"""
Backend Server for LLM Chatbot with Context Management.
Uses aiohttp to serve async REST endpoints and modern web client.
Run with: python server.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Configure UTF-8 stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from aiohttp import web

from config import settings, SYSTEM_PROMPTS
from llm_client import (
    LLMClient,
    LLMException,
    LLMConfigError,
    LLMAuthError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMAPIError,
)
from context_manager import ContextManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

client = LLMClient(settings)
context_mgr = ContextManager(
    strategy=settings.context_strategy,
    max_history_messages=settings.max_history_messages,
    llm_client=client,
)


# ==============================================================================
# REST Endpoints
# ==============================================================================

async def health_handler(request: web.Request) -> web.Response:
    """Returns provider status and readiness info."""
    status = client.get_health_status()
    return web.json_response(status)


async def config_handler(request: web.Request) -> web.Response:
    """Returns current configuration options and personas."""
    return web.json_response({
        "provider": client.provider,
        "model": settings.gemini_model if client.provider == "gemini" else settings.openai_model,
        "active_persona": settings.persona,
        "active_strategy": context_mgr.strategy,
        "max_history_messages": context_mgr.max_history_messages,
        "available_personas": list(SYSTEM_PROMPTS.keys()),
        "system_prompt": settings.system_prompt,
    })


async def update_config_handler(request: web.Request) -> web.Response:
    """Updates runtime configuration (strategy, persona, max_history)."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": True, "message": "Invalid JSON body"}, status=400)

    strategy = data.get("strategy")
    if strategy in ("sliding_window", "summarization"):
        context_mgr.strategy = strategy

    persona = data.get("persona")
    if persona and persona in SYSTEM_PROMPTS:
        settings.persona = persona
        settings.system_prompt = SYSTEM_PROMPTS[persona]

    if "max_history_messages" in data:
        try:
            val = int(data["max_history_messages"])
            if val >= 2:
                context_mgr.max_history_messages = val
        except ValueError:
            pass

    return web.json_response({
        "status": "updated",
        "active_persona": settings.persona,
        "active_strategy": context_mgr.strategy,
        "max_history_messages": context_mgr.max_history_messages,
    })


async def history_handler(request: web.Request) -> web.Response:
    """Returns full message history for a given session."""
    session_id = request.query.get("session_id", "default")
    history = context_mgr.get_history(session_id)
    session = context_mgr.get_or_create_session(session_id)
    return web.json_response({
        "session_id": session_id,
        "messages": history,
        "running_summary": session.running_summary,
        "total_messages": len(history),
    })


async def reset_handler(request: web.Request) -> web.Response:
    """Clears history and memory for a given session."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    session_id = data.get("session_id", "default")
    context_mgr.clear_session(session_id)
    logger.info("Session '%s' memory cleared.", session_id)
    return web.json_response({
        "status": "reset",
        "session_id": session_id,
        "message": "Conversation history and summary have been reset."
    })


async def chat_handler(request: web.Request) -> web.Response:
    """
    Main chat endpoint. Receives user message, runs context window management,
    calls the LLM API, and returns assistant reply with debug metadata.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": True, "message": "Invalid JSON payload."}, status=400)

    user_text = body.get("message", "").strip()
    session_id = body.get("session_id", "default").strip() or "default"
    override_strategy = body.get("strategy")

    if not user_text:
        return web.json_response({"error": True, "message": "Message cannot be empty."}, status=400)

    # 1. Add user message to conversation history
    context_mgr.add_message(session_id, "user", user_text)

    try:
        # 2. Context Window Management: prepare payload (truncation or running summary)
        payload_messages, metadata = await context_mgr.prepare_context_async(
            session_id=session_id,
            override_strategy=override_strategy,
        )

        logger.info(
            "Session '%s': Sending %d messages to %s (Total in history: %d, Evicted: %d, Strategy: %s)",
            session_id,
            len(payload_messages),
            client.provider,
            metadata["total_messages"],
            metadata["evicted_messages"],
            metadata["strategy"],
        )

        # 3. Call LLM with the managed context payload & system prompt
        assistant_reply = await client.generate_response(
            messages=payload_messages,
            system_instruction=settings.system_prompt,
        )

        # 4. Record assistant reply in history
        context_mgr.add_message(session_id, "assistant", assistant_reply)

        # Update metadata with final count
        metadata["turns"] = (metadata["total_messages"] + 1) // 2

        return web.json_response({
            "response": assistant_reply,
            "session_id": session_id,
            "metrics": metadata,
            "persona": settings.persona,
        })

    except LLMConfigError as e:
        logger.error("Configuration error: %s", e)
        return web.json_response({
            "error": True,
            "error_type": "ConfigError",
            "message": str(e),
            "suggestion": "Check your .env file and ensure GEMINI_API_KEY or OPENAI_API_KEY is configured."
        }, status=400)

    except LLMAuthError as e:
        logger.error("Authentication error: %s", e)
        return web.json_response({
            "error": True,
            "error_type": "AuthError",
            "message": str(e),
            "suggestion": "Your API key was rejected by the provider. Please verify it in .env."
        }, status=401)

    except LLMRateLimitError as e:
        logger.warning("Rate limit hit: %s", e)
        return web.json_response({
            "error": True,
            "error_type": "RateLimitError",
            "message": str(e),
            "suggestion": "Wait a moment for rate limits to reset, or check credit quotas."
        }, status=429)

    except LLMConnectionError as e:
        logger.error("Connection error: %s", e)
        return web.json_response({
            "error": True,
            "error_type": "ConnectionError",
            "message": str(e),
            "suggestion": "Ensure this machine has internet connectivity and can reach the LLM API endpoints."
        }, status=503)

    except LLMAPIError as e:
        logger.error("LLM API error: %s", e)
        return web.json_response({
            "error": True,
            "error_type": "APIError",
            "message": str(e),
            "suggestion": "The LLM provider returned an unexpected error. Please retry."
        }, status=502)

    except Exception as e:
        logger.exception("Unexpected server error: %s", e)
        return web.json_response({
            "error": True,
            "error_type": "ServerError",
            "message": f"Internal server error: {e}",
        }, status=500)


# ==============================================================================
# Application Factory
# ==============================================================================

def create_app() -> web.Application:
    app = web.Application()

    # API routes
    app.router.add_get("/api/health", health_handler)
    app.router.add_get("/api/config", config_handler)
    app.router.add_post("/api/config", update_config_handler)
    app.router.add_get("/api/history", history_handler)
    app.router.add_post("/api/reset", reset_handler)
    app.router.add_post("/api/chat", chat_handler)

    # Static file routes (custom to prevent Windows Path.resolve WinError 5)
    async def index_handler(request: web.Request) -> web.Response:
        index_file = STATIC_DIR / "index.html"
        if index_file.is_file():
            content = index_file.read_text(encoding="utf-8")
            return web.Response(text=content, content_type="text/html")
        return web.Response(status=404, text="index.html not found")

    async def static_file_handler(request: web.Request) -> web.Response:
        filename = request.match_info.get("filename", "")
        # Prevent directory traversal
        safe_filename = Path(filename).name
        target = STATIC_DIR / safe_filename
        if target.is_file():
            content = target.read_text(encoding="utf-8")
            content_type = "text/plain"
            if safe_filename.endswith(".css"):
                content_type = "text/css"
            elif safe_filename.endswith(".js"):
                content_type = "application/javascript"
            elif safe_filename.endswith(".html"):
                content_type = "text/html"
            return web.Response(text=content, content_type=content_type)
        return web.Response(status=404, text=f"Static file '{filename}' not found")

    app.router.add_get("/", index_handler)
    app.router.add_get("/static/{filename}", static_file_handler)

    return app


if __name__ == "__main__":
    host = settings.host
    port = settings.port
    print("=" * 65)
    print(f">> Starting Archie LLM Chat Server on http://{host}:{port}")
    print(f">> Provider: {client.provider.upper()} | Strategy: {context_mgr.strategy}")
    print(f">> System Persona: {settings.persona}")
    print("=" * 65)
    app = create_app()
    web.run_app(app, host=host, port=port)
