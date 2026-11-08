"""
Unified LLM Client module.
Provides an abstracted, resilient interface to Google Gemini and OpenAI APIs
with complete error handling, timeouts, and multi-turn message formatting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

from config import Settings, settings

logger = logging.getLogger("llm_client")


# ==============================================================================
# Custom Exceptions for Granular Error Handling
# ==============================================================================

class LLMException(Exception):
    """Base exception for all LLM client failures."""
    pass


class LLMConfigError(LLMException):
    """Raised when API keys or configurations are missing or invalid."""
    pass


class LLMAuthError(LLMException):
    """Raised when authentication fails (HTTP 401/403: bad API key)."""
    pass


class LLMRateLimitError(LLMException):
    """Raised when API rate limits or quota limits are exceeded (HTTP 429)."""
    pass


class LLMConnectionError(LLMException):
    """Raised when network, DNS, or socket connection fails."""
    pass


class LLMAPIError(LLMException):
    """Raised when the upstream provider returns an error response."""
    pass


# ==============================================================================
# LLM Client Implementation
# ==============================================================================

class LLMClient:
    """
    Unified client supporting Google Gemini and OpenAI with consistent interface.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or settings
        self.provider = self.config.provider
        self.timeout = 30.0  # seconds

    def _get_active_api_key(self) -> str:
        if self.provider == "gemini":
            key = self.config.gemini_api_key
            if not key:
                raise LLMConfigError(
                    "Missing Gemini API Key. Please set GEMINI_API_KEY in your .env file. "
                    "You can generate a free key at https://aistudio.google.com/"
                )
            return key
        elif self.provider == "openai":
            key = self.config.openai_api_key
            if not key:
                raise LLMConfigError(
                    "Missing OpenAI API Key. Please set OPENAI_API_KEY in your .env file. "
                    "You can generate a key at https://platform.openai.com/api-keys"
                )
            return key
        else:
            raise LLMConfigError(f"Unsupported provider: '{self.provider}'")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Asynchronously sends conversation history to the configured LLM API.
        
        Args:
            messages: List of message dicts with keys 'role' ('user'|'assistant') and 'content'.
            system_instruction: The defined persona and behavioral prompt.
            temperature: Creativity control (0.0 = deterministic, 1.0 = creative).
            
        Returns:
            The text response from the LLM.
        """
        api_key = self._get_active_api_key()
        instruction = system_instruction or self.config.system_prompt

        if self.provider == "gemini":
            return await self._call_gemini_async(messages, instruction, api_key, temperature)
        elif self.provider == "openai":
            return await self._call_openai_async(messages, instruction, api_key, temperature)
        else:
            raise LLMConfigError(f"Unsupported provider: {self.provider}")

    def generate_response_sync(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Synchronous counterpart for CLI scripts or one-off tests.
        """
        api_key = self._get_active_api_key()
        instruction = system_instruction or self.config.system_prompt

        if self.provider == "gemini":
            return self._call_gemini_sync(messages, instruction, api_key, temperature)
        elif self.provider == "openai":
            return self._call_openai_sync(messages, instruction, api_key, temperature)
        else:
            raise LLMConfigError(f"Unsupported provider: {self.provider}")

    # --------------------------------------------------------------------------
    # Google Gemini Implementation
    # --------------------------------------------------------------------------

    def _call_gemini_sync(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str,
        api_key: str,
        temperature: float,
    ) -> str:
        try:
            from google import genai
            from google.genai import types
            from google.genai.errors import APIError

            client = genai.Client(api_key=api_key)

            # Convert message history to Gemini contents
            gemini_contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            )

            response = client.models.generate_content(
                model=self.config.gemini_model,
                contents=gemini_contents,
                config=config,
            )

            if not response.text:
                raise LLMAPIError("Gemini returned an empty response. The content may have triggered safety filters.")

            return response.text

        except ImportError:
            # Fallback to direct HTTP REST if SDK is somehow missing
            return self._call_gemini_rest_sync(messages, system_instruction, api_key, temperature)
        except Exception as e:
            self._handle_gemini_error(e)

    async def _call_gemini_async(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str,
        api_key: str,
        temperature: float,
    ) -> str:
        # Use httpx async client for non-blocking I/O
        return await self._call_gemini_rest_async(messages, system_instruction, api_key, temperature)

    async def _call_gemini_rest_async(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str,
        api_key: str,
        temperature: float,
    ) -> str:
        model = self.config.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload: Dict[str, Any] = {
            "contents": contents,
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "temperature": temperature,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 400:
                    raise LLMAuthError(f"Bad Request / Invalid API Key ({res.status_code}): {res.text}")
                elif res.status_code in (401, 403):
                    raise LLMAuthError(f"Authentication failed ({res.status_code}). Verify your GEMINI_API_KEY.")
                elif res.status_code == 429:
                    raise LLMRateLimitError("Gemini rate limit exceeded (HTTP 429). Please wait a moment.")
                elif res.status_code >= 500:
                    raise LLMAPIError(f"Gemini server error ({res.status_code}). Please retry.")

                res.raise_for_status()
                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise LLMAPIError("No response candidates returned by Gemini.")
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise LLMAPIError("Empty content parts in Gemini response.")
                return parts[0].get("text", "")

        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Failed to connect to Google Gemini API: {e}. Check your network connection."
            ) from e
        except httpx.TimeoutException as e:
            raise LLMConnectionError(
                f"Request to Google Gemini API timed out after {self.timeout}s."
            ) from e
        except LLMException:
            raise
        except Exception as e:
            self._handle_gemini_error(e)

    def _call_gemini_rest_sync(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str,
        api_key: str,
        temperature: float,
    ) -> str:
        model = self.config.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "contents": contents,
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"temperature": temperature}
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)
                if res.status_code in (401, 403):
                    raise LLMAuthError(f"Authentication failed ({res.status_code}). Verify your GEMINI_API_KEY.")
                elif res.status_code == 429:
                    raise LLMRateLimitError("Gemini rate limit exceeded (HTTP 429).")
                elif res.status_code >= 500:
                    raise LLMAPIError(f"Gemini server error ({res.status_code}).")
                res.raise_for_status()
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to connect to Google Gemini API: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"Request to Google Gemini API timed out after {self.timeout}s.") from e
        except Exception as e:
            self._handle_gemini_error(e)

    def _handle_gemini_error(self, err: Exception):
        err_msg = str(err)
        if "API_KEY_INVALID" in err_msg or "403" in err_msg or "401" in err_msg:
            raise LLMAuthError("Invalid Gemini API Key. Please verify your GEMINI_API_KEY in .env.") from err
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            raise LLMRateLimitError("Gemini API quota exceeded or rate limit reached (HTTP 429).") from err
        if "ConnectError" in type(err).__name__ or "getaddrinfo failed" in err_msg:
            raise LLMConnectionError("Network connection failure when reaching Gemini API.") from err
        raise LLMAPIError(f"Gemini API Error: {err}") from err

    # --------------------------------------------------------------------------
    # OpenAI Implementation
    # --------------------------------------------------------------------------

    def _format_openai_messages(self, messages: List[Dict[str, str]], system_instruction: str) -> List[Dict[str, str]]:
        formatted = [{"role": "system", "content": system_instruction}]
        for msg in messages:
            role = "assistant" if msg["role"] in ("assistant", "model") else "user"
            formatted.append({"role": role, "content": msg["content"]})
        return formatted

    async def _call_openai_async(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str,
        api_key: str,
        temperature: float,
    ) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.openai_model,
            "messages": self._format_openai_messages(messages, system_instruction),
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, headers=headers, json=payload)
                return self._parse_openai_response(res)
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to connect to OpenAI API: {e}. Check your internet connection.") from e
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"Request to OpenAI API timed out after {self.timeout}s.") from e
        except LLMException:
            raise
        except Exception as e:
            raise LLMAPIError(f"Unexpected OpenAI error: {e}") from e

    def _call_openai_sync(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str,
        api_key: str,
        temperature: float,
    ) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.openai_model,
            "messages": self._format_openai_messages(messages, system_instruction),
            "temperature": temperature,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, headers=headers, json=payload)
                return self._parse_openai_response(res)
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to connect to OpenAI API: {e}.") from e
        except httpx.TimeoutException as e:
            raise LLMConnectionError(f"Request to OpenAI API timed out after {self.timeout}s.") from e
        except LLMException:
            raise
        except Exception as e:
            raise LLMAPIError(f"Unexpected OpenAI error: {e}") from e

    def _parse_openai_response(self, res: httpx.Response) -> str:
        if res.status_code in (401, 403):
            raise LLMAuthError(f"Authentication failed ({res.status_code}): Invalid OPENAI_API_KEY.")
        if res.status_code == 429:
            raise LLMRateLimitError(
                "OpenAI rate limit or credit quota exceeded (HTTP 429). Check your billing / quota at platform.openai.com."
            )
        if res.status_code >= 500:
            raise LLMAPIError(f"OpenAI service outage / server error ({res.status_code}).")

        try:
            res.raise_for_status()
            data = res.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMAPIError("No completion choices returned by OpenAI.")
            return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            if isinstance(e, LLMException):
                raise
            raise LLMAPIError(f"Failed to parse OpenAI response: {res.text}") from e

    def get_health_status(self) -> Dict[str, Any]:
        """Returns non-sensitive health metadata for UI and monitoring."""
        has_gemini = bool(self.config.gemini_api_key)
        has_openai = bool(self.config.openai_api_key)
        active_key_ready = (self.provider == "gemini" and has_gemini) or (self.provider == "openai" and has_openai)

        return {
            "status": "configured" if active_key_ready else "missing_api_key",
            "provider": self.provider,
            "model": self.config.gemini_model if self.provider == "gemini" else self.config.openai_model,
            "has_gemini_key": has_gemini,
            "has_openai_key": has_openai,
            "active_key_ready": active_key_ready,
            "context_strategy": self.config.context_strategy,
            "max_history_messages": self.config.max_history_messages,
            "persona": self.config.persona,
        }
