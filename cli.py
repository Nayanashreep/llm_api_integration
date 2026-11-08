"""
Interactive Terminal CLI Interface for LLM Chatbot.
Supports rich Markdown rendering, real-time context management,
strategy switching, and interactive debug inspection.
Run with: python cli.py
"""

from __future__ import annotations

import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt

from config import settings, SYSTEM_PROMPTS
from llm_client import (
    LLMClient,
    LLMConfigError,
    LLMAuthError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMAPIError,
)
from context_manager import ContextManager

console = Console()


def render_banner(client: LLMClient, cm: ContextManager):
    table = Table(title="🤖 Archie LLM Assistant (Active Config)", show_header=False, border_style="cyan")
    table.add_row("Provider", f"[bold green]{client.provider.upper()}[/bold green]")
    model_name = settings.gemini_model if client.provider == "gemini" else settings.openai_model
    table.add_row("Model", f"[yellow]{model_name}[/yellow]")
    table.add_row("Persona", f"[magenta]{settings.persona}[/magenta]")
    table.add_row("Context Strategy", f"[cyan]{cm.strategy}[/cyan] (Max {cm.max_history_messages} msgs)")
    table.add_row("Commands", "[dim]/help, /context, /summary, /strategy, /persona, /clear, /quit[/dim]")
    console.print(table)


def render_help():
    table = Table(title="CLI Commands", border_style="dim")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("/help", "Show this help table")
    table.add_row("/context", "Inspect active window, payload, and eviction stats")
    table.add_row("/summary", "View the current running memory summary")
    table.add_row("/history", "Display full conversation history for this session")
    table.add_row("/strategy <name>", "Switch strategy: 'sliding_window' or 'summarization'")
    table.add_row("/persona <name>", "Switch persona: architect, socratic, reviewer, companion")
    table.add_row("/clear", "Reset current conversation memory and start fresh")
    table.add_row("/quit or /exit", "Exit the chatbot")
    console.print(table)


def main():
    console.clear()
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold yellow]       🧠 Multi-Turn LLM Chatbot with Context Management[/bold yellow]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════════════[/bold cyan]\n")

    client = LLMClient(settings)
    cm = ContextManager(
        strategy=settings.context_strategy,
        max_history_messages=settings.max_history_messages,
        llm_client=client,
    )
    session_id = "cli_session"

    render_banner(client, cm)
    console.print("\n[dim]Type your message below. Type /help for command list.[/dim]\n")

    turn = 1
    while True:
        try:
            user_input = Prompt.ask(f"[bold green]Turn {turn} [You][/bold green]").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    console.print("[yellow]Exiting chat. Have a great day![/yellow]")
                    break

                elif cmd == "/help":
                    render_help()
                    continue

                elif cmd == "/clear" or cmd == "/reset":
                    cm.clear_session(session_id)
                    turn = 1
                    console.print("[bold yellow]🧹 Conversation memory reset to zero.[/bold yellow]\n")
                    continue

                elif cmd == "/history":
                    history = cm.get_history(session_id)
                    console.print(f"\n[bold]Full Session History ({len(history)} messages):[/bold]")
                    for idx, m in enumerate(history, 1):
                        role_color = "green" if m["role"] == "user" else "cyan"
                        console.print(f"[{role_color}][{idx}] {m['role'].upper()}:[/{role_color}] {m['content']}")
                    console.print()
                    continue

                elif cmd in ("/context", "/debug"):
                    payload, meta = cm.prepare_context_sync(session_id)
                    dbg_table = Table(title="Context Window Debug Inspector", border_style="yellow")
                    dbg_table.add_column("Property", style="bold")
                    dbg_table.add_column("Value")
                    dbg_table.add_row("Total Messages Recorded", str(meta["total_messages"]))
                    dbg_table.add_row("Active Window Messages", str(meta["active_window_messages"]))
                    dbg_table.add_row("Evicted / Truncated Messages", str(meta["evicted_messages"]))
                    dbg_table.add_row("Active Strategy", meta["strategy"])
                    dbg_table.add_row("Has Running Summary", str(meta["has_summary"]))
                    if meta["running_summary"]:
                        dbg_table.add_row("Current Running Summary", meta["running_summary"])
                    console.print(dbg_table)
                    continue

                elif cmd == "/summary":
                    session = cm.get_or_create_session(session_id)
                    if session.running_summary:
                        console.print(Panel(session.running_summary, title="🧠 Active Running Summary", border_style="cyan"))
                    else:
                        console.print("[dim]No running summary yet (conversation has not exceeded capacity).[/dim]")
                    continue

                elif cmd == "/strategy":
                    if arg in ("sliding_window", "summarization"):
                        cm.strategy = arg
                        console.print(f"[green]Context strategy changed to: [bold]{arg}[/bold][/green]")
                    else:
                        console.print("[red]Usage: /strategy sliding_window | summarization[/red]")
                    continue

                elif cmd == "/persona":
                    if arg in SYSTEM_PROMPTS:
                        settings.persona = arg
                        settings.system_prompt = SYSTEM_PROMPTS[arg]
                        console.print(f"[green]Persona updated to: [bold]{arg}[/bold][/green]")
                    else:
                        valid = ", ".join(SYSTEM_PROMPTS.keys())
                        console.print(f"[red]Invalid persona. Valid options: {valid}[/red]")
                    continue

                else:
                    console.print(f"[red]Unknown command '{cmd}'. Type /help for available commands.[/red]")
                    continue

            # Add user message to conversation history
            cm.add_message(session_id, "user", user_input)

            # Prepare context based on active strategy
            with console.status("[bold cyan]Thinking and managing context...[/bold cyan]", spinner="dots"):
                payload_messages, meta = cm.prepare_context_sync(session_id)

                # Send context-managed payload to LLM
                response_text = client.generate_response_sync(
                    messages=payload_messages,
                    system_instruction=settings.system_prompt,
                )

            # Record assistant response in history
            cm.add_message(session_id, "assistant", response_text)

            # Render response
            badge = f"Context: {meta['active_window_messages']}/{meta['total_messages']} msgs"
            if meta["has_summary"]:
                badge += " | [bold yellow]+Memory Summary[/bold yellow]"

            console.print(Panel(
                Markdown(response_text),
                title=f"🤖 Archie ({settings.persona.capitalize()})",
                subtitle=f"[dim]{badge}[/dim]",
                border_style="cyan",
            ))
            console.print()
            turn += 1

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
            break
        except LLMConfigError as e:
            console.print(Panel(str(e), title="❌ Configuration Error", border_style="red"))
        except LLMAuthError as e:
            console.print(Panel(str(e), title="❌ Authentication Error", border_style="red"))
        except LLMRateLimitError as e:
            console.print(Panel(str(e), title="⚠️ Rate Limit", border_style="yellow"))
        except LLMConnectionError as e:
            console.print(Panel(str(e), title="🌐 Connection Error", border_style="red"))
        except LLMAPIError as e:
            console.print(Panel(str(e), title="⚠️ API Error", border_style="red"))
        except Exception as e:
            console.print(Panel(f"Unexpected error: {e}", title="❌ Error", border_style="red"))


if __name__ == "__main__":
    main()
