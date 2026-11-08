"""
Configuration module for the LLM Chatbot.
Handles loading environment variables, provider settings, and system prompt personas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Literal
from dotenv import load_dotenv

# Load .env file if available
load_dotenv()

# System prompt personas with distinct personalities and constraints
SYSTEM_PROMPTS: Dict[str, str] = {
    "architect": (
        "You are Archie, a pragmatic Senior Software Architect and Pair Programmer. "
        "Your personality is analytical, encouraging, and razor-sharp. "
        "Core directives:\n"
        "1. Never give lazy or generic answers; reason carefully through trade-offs.\n"
        "2. Break down complex design decisions with clear pros/cons.\n"
        "3. Provide clean, well-commented, production-ready code examples when asked.\n"
        "4. Challenge assumptions constructively and always highlight edge cases, security, or scalability.\n"
        "5. Keep responses concise and focused without unnecessary fluff."
    ),
    "socratic": (
        "You are Socrates, a thoughtful technical tutor. "
        "Instead of immediately giving away full solutions, guide the user through "
        "thought-provoking questions, analogies, and hints so they learn the underlying principles."
    ),
    "reviewer": (
        "You are a strict, detail-oriented Staff Security & Performance Code Reviewer. "
        "Identify potential bugs, race conditions, memory leaks, security vulnerabilities, "
        "and suggest idiomatic, high-performance improvements."
    ),
    "companion": (
        "You are Nova, an enthusiastic and witty AI technology companion. "
        "You explain complex topics using vivid analogies, infectious curiosity, "
        "and clear, accessible language."
    ),
}

DEFAULT_PERSONA = "architect"


@dataclass
class Settings:
    provider: Literal["gemini", "openai"]
    gemini_api_key: str
    gemini_model: str
    openai_api_key: str
    openai_model: str
    context_strategy: Literal["sliding_window", "summarization"]
    max_history_messages: int
    persona: str
    system_prompt: str
    host: str
    port: int

    @classmethod
    def load(cls) -> Settings:
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        configured_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()

        # Auto-detect provider if "auto"
        if configured_provider == "auto":
            if gemini_key:
                provider = "gemini"
            elif openai_key:
                provider = "openai"
            else:
                provider = "gemini"  # default fallback
        elif configured_provider in ("gemini", "google"):
            provider = "gemini"
        elif configured_provider in ("openai", "gpt"):
            provider = "openai"
        else:
            provider = "gemini"

        persona_key = os.getenv("PERSONA", DEFAULT_PERSONA).lower()
        custom_prompt = os.getenv("SYSTEM_PROMPT", "").strip()
        system_prompt = custom_prompt if custom_prompt else SYSTEM_PROMPTS.get(persona_key, SYSTEM_PROMPTS[DEFAULT_PERSONA])

        strategy = os.getenv("CONTEXT_STRATEGY", "summarization").strip().lower()
        if strategy not in ("sliding_window", "summarization"):
            strategy = "summarization"

        try:
            max_messages = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
        except ValueError:
            max_messages = 6

        return cls(
            provider=provider,
            gemini_api_key=gemini_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            openai_api_key=openai_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            context_strategy=strategy,
            max_history_messages=max_messages,
            persona=persona_key,
            system_prompt=system_prompt,
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8000")),
        )


settings = Settings.load()
