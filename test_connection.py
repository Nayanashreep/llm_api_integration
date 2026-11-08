"""
Minimal test connection script (Step 2 of assignment).
Sends a single message to verify the LLM API connection and prints the response.
Run with: python test_connection.py
"""

from __future__ import annotations

import sys

# Configure UTF-8 stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import settings
from llm_client import (
    LLMClient,
    LLMConfigError,
    LLMAuthError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMAPIError,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None


def print_message(title: str, text: str, style: str = "white"):
    if console:
        console.print(Panel(text, title=title, border_style=style))
    else:
        print(f"\n--- {title} ---\n{text}\n" + "-" * 30)


def main():
    print("=" * 60)
    print(">> LLM API Connectivity Test")
    print("=" * 60)
    print(f"Provider : {settings.provider.upper()}")
    model = settings.gemini_model if settings.provider == "gemini" else settings.openai_model
    print(f"Model    : {model}")
    print(f"Persona  : {settings.persona}")
    print("=" * 60)

    client = LLMClient(settings)
    test_prompt = "Say hello, briefly state your persona, and confirm that the API connection is active and responsive."

    print("\n>> Sending single-turn verification message to LLM API...")
    messages = [{"role": "user", "content": test_prompt}]

    try:
        response = client.generate_response_sync(messages)
        print("\n[OK] API Connection Successful!\n")
        print_message("LLM Response", response, style="green")
        print("\n[SUCCESS] Connection verified! You are ready to run the chatbot.")
        print("To start the web app:  python server.py")
        print("To start the terminal: python cli.py\n")
        sys.exit(0)

    except LLMConfigError as e:
        print("\n[ERROR] Configuration Error:")
        print_message("Missing Key", str(e), style="red")
        print("Tip: Copy .env.example to .env and fill in your API key.")
        sys.exit(1)

    except LLMAuthError as e:
        print("\n[ERROR] Authentication Error:")
        print_message("Invalid Key", str(e), style="red")
        print("Tip: Double check the API key you entered in .env.")
        sys.exit(1)

    except LLMRateLimitError as e:
        print("\n[WARNING] Rate Limit Exceeded:")
        print_message("Quota / Rate Limit", str(e), style="yellow")
        print("Tip: Wait a few seconds or check your account credits.")
        sys.exit(1)

    except LLMConnectionError as e:
        print("\n[ERROR] Network / Connection Error:")
        print_message("Connection Failed", str(e), style="red")
        print("Tip: Ensure your machine has active internet access and can reach the API.")
        sys.exit(1)

    except LLMAPIError as e:
        print("\n[ERROR] Provider API Error:")
        print_message("API Error", str(e), style="red")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
