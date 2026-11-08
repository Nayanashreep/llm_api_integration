"""
Context Window Manager Module.
Implements multi-turn history tracking and two context management strategies:
1. Sliding Window Truncation (keeps the last N messages)
2. Running Summarization + Sliding Window (compresses evicted turns into running memory)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger("context_manager")


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class ConversationSession:
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    running_summary: str = ""
    last_summarized_index: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ContextManager:
    """
    Manages conversational memory, sessions, and context window compaction.
    """

    def __init__(
        self,
        strategy: Literal["sliding_window", "summarization"] = "summarization",
        max_history_messages: int = 6,
        llm_client: Optional[Any] = None,
    ):
        self.strategy = strategy
        self.max_history_messages = max(2, max_history_messages)
        self.llm_client = llm_client
        self.sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: str = "default") -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: Literal["user", "assistant"], content: str) -> ChatMessage:
        session = self.get_or_create_session(session_id)
        msg = ChatMessage(role=role, content=content.strip())
        session.messages.append(msg)
        session.updated_at = time.time()
        return msg

    def clear_session(self, session_id: str = "default") -> None:
        if session_id in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)

    def get_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        session = self.get_or_create_session(session_id)
        return [msg.to_dict() for msg in session.messages]

    def _generate_extractive_summary(self, messages_to_summarize: List[ChatMessage]) -> str:
        """Deterministic fallback summary if LLM call is unavailable or fails."""
        bullets = []
        for m in messages_to_summarize:
            first_line = m.content.strip().split("\n")[0]
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            speaker = "User mentioned" if m.role == "user" else "Assistant advised"
            bullets.append(f"- {speaker}: {first_line}")
        return "\n".join(bullets)

    async def _update_running_summary_async(
        self, session: ConversationSession, evicted_messages: List[ChatMessage]
    ) -> None:
        """Updates the running summary with newly evicted messages asynchronously."""
        if not evicted_messages:
            return

        evicted_transcript = "\n".join(
            f"{m.role.capitalize()}: {m.content}" for m in evicted_messages
        )

        prompt = (
            "You are a conversation memory condenser. "
            "Update the running summary of this conversation with the new dialogue points below.\n"
            "Keep key facts, user preferences, code context, goals, and constraints.\n"
            "Produce 2-4 concise bullet points.\n\n"
        )
        if session.running_summary:
            prompt += f"CURRENT SUMMARY:\n{session.running_summary}\n\n"

        prompt += f"NEW EVICTED CONVERSATION:\n{evicted_transcript}\n\nUPDATED CONCISE SUMMARY:"

        if self.llm_client:
            try:
                summary_response = await self.llm_client.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    system_instruction="You are a strict, objective factual summarizer. Output only summary bullet points.",
                    temperature=0.2,
                )
                session.running_summary = summary_response.strip()
                return
            except Exception as e:
                logger.warning("LLM summarization failed (%s), using extractive fallback.", e)

        # Fallback if LLM call fails
        fallback = self._generate_extractive_summary(evicted_messages)
        if session.running_summary:
            session.running_summary += f"\n{fallback}"
        else:
            session.running_summary = fallback

    def _update_running_summary_sync(
        self, session: ConversationSession, evicted_messages: List[ChatMessage]
    ) -> None:
        """Synchronous version for CLI usage."""
        if not evicted_messages:
            return

        evicted_transcript = "\n".join(
            f"{m.role.capitalize()}: {m.content}" for m in evicted_messages
        )

        prompt = (
            "You are a conversation memory condenser. "
            "Update the running summary of this conversation with the new dialogue points below.\n"
            "Keep key facts, user preferences, code context, goals, and constraints.\n"
            "Produce 2-4 concise bullet points.\n\n"
        )
        if session.running_summary:
            prompt += f"CURRENT SUMMARY:\n{session.running_summary}\n\n"

        prompt += f"NEW EVICTED CONVERSATION:\n{evicted_transcript}\n\nUPDATED CONCISE SUMMARY:"

        if self.llm_client:
            try:
                summary_response = self.llm_client.generate_response_sync(
                    messages=[{"role": "user", "content": prompt}],
                    system_instruction="You are a strict, objective factual summarizer. Output only summary bullet points.",
                    temperature=0.2,
                )
                session.running_summary = summary_response.strip()
                return
            except Exception as e:
                logger.warning("LLM summarization failed (%s), using extractive fallback.", e)

        fallback = self._generate_extractive_summary(evicted_messages)
        if session.running_summary:
            session.running_summary += f"\n{fallback}"
        else:
            session.running_summary = fallback

    async def prepare_context_async(
        self,
        session_id: str = "default",
        override_strategy: Optional[Literal["sliding_window", "summarization"]] = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Prepares the message payload for the LLM based on the active strategy.
        Returns:
            (payload_messages, context_metadata)
        """
        session = self.get_or_create_session(session_id)
        total_messages = len(session.messages)
        strategy = override_strategy or self.strategy

        # Case 1: Under capacity
        if total_messages <= self.max_history_messages:
            payload = [{"role": m.role, "content": m.content} for m in session.messages]
            metadata = {
                "session_id": session_id,
                "strategy": strategy,
                "total_messages": total_messages,
                "active_window_messages": total_messages,
                "evicted_messages": 0,
                "has_summary": bool(session.running_summary),
                "running_summary": session.running_summary,
            }
            return payload, metadata

        # Case 2: Above capacity -> Apply strategy
        active_slice = session.messages[-self.max_history_messages :]
        evicted_count = total_messages - self.max_history_messages

        if strategy == "sliding_window":
            # Pure truncation: keep only the most recent N messages
            payload = [{"role": m.role, "content": m.content} for m in active_slice]
            metadata = {
                "session_id": session_id,
                "strategy": "sliding_window",
                "total_messages": total_messages,
                "active_window_messages": len(active_slice),
                "evicted_messages": evicted_count,
                "has_summary": False,
                "running_summary": "",
            }
            return payload, metadata

        elif strategy == "summarization":
            # Running summarization: summarize newly evicted messages
            cutoff_index = total_messages - self.max_history_messages
            newly_evicted = session.messages[session.last_summarized_index : cutoff_index]

            if newly_evicted:
                await self._update_running_summary_async(session, newly_evicted)
                session.last_summarized_index = cutoff_index

            # Inject memory context before the active window
            payload: List[Dict[str, str]] = []
            if session.running_summary:
                payload.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM MEMORY CONTEXT: Summary of earlier conversation turns]\n"
                        f"{session.running_summary}\n"
                        f"[END MEMORY CONTEXT]"
                    ),
                })
                payload.append({
                    "role": "assistant",
                    "content": (
                        "I have absorbed the earlier conversation memory and context. "
                        "Continuing our active discussion now."
                    ),
                })

            for m in active_slice:
                payload.append({"role": m.role, "content": m.content})

            metadata = {
                "session_id": session_id,
                "strategy": "summarization",
                "total_messages": total_messages,
                "active_window_messages": len(active_slice),
                "evicted_messages": evicted_count,
                "has_summary": bool(session.running_summary),
                "running_summary": session.running_summary,
            }
            return payload, metadata

        else:
            raise ValueError(f"Unknown context strategy: {strategy}")

    def prepare_context_sync(
        self,
        session_id: str = "default",
        override_strategy: Optional[Literal["sliding_window", "summarization"]] = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Synchronous version for CLI."""
        session = self.get_or_create_session(session_id)
        total_messages = len(session.messages)
        strategy = override_strategy or self.strategy

        if total_messages <= self.max_history_messages:
            payload = [{"role": m.role, "content": m.content} for m in session.messages]
            metadata = {
                "session_id": session_id,
                "strategy": strategy,
                "total_messages": total_messages,
                "active_window_messages": total_messages,
                "evicted_messages": 0,
                "has_summary": bool(session.running_summary),
                "running_summary": session.running_summary,
            }
            return payload, metadata

        active_slice = session.messages[-self.max_history_messages :]
        evicted_count = total_messages - self.max_history_messages

        if strategy == "sliding_window":
            payload = [{"role": m.role, "content": m.content} for m in active_slice]
            metadata = {
                "session_id": session_id,
                "strategy": "sliding_window",
                "total_messages": total_messages,
                "active_window_messages": len(active_slice),
                "evicted_messages": evicted_count,
                "has_summary": False,
                "running_summary": "",
            }
            return payload, metadata

        elif strategy == "summarization":
            cutoff_index = total_messages - self.max_history_messages
            newly_evicted = session.messages[session.last_summarized_index : cutoff_index]

            if newly_evicted:
                self._update_running_summary_sync(session, newly_evicted)
                session.last_summarized_index = cutoff_index

            payload: List[Dict[str, str]] = []
            if session.running_summary:
                payload.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM MEMORY CONTEXT: Summary of earlier conversation turns]\n"
                        f"{session.running_summary}\n"
                        f"[END MEMORY CONTEXT]"
                    ),
                })
                payload.append({
                    "role": "assistant",
                    "content": (
                        "I have absorbed the earlier conversation memory and context. "
                        "Continuing our active discussion now."
                    ),
                })

            for m in active_slice:
                payload.append({"role": m.role, "content": m.content})

            metadata = {
                "session_id": session_id,
                "strategy": "summarization",
                "total_messages": total_messages,
                "active_window_messages": len(active_slice),
                "evicted_messages": evicted_count,
                "has_summary": bool(session.running_summary),
                "running_summary": session.running_summary,
            }
            return payload, metadata
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
