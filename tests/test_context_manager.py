"""
Unit tests for ContextManager and LLM Client components.
Validates multi-turn conversation memory, sliding window truncation, and running summarization.
Can be executed with standard library: python -m unittest discover tests
"""

import unittest
from context_manager import ContextManager, ChatMessage
from llm_client import (
    LLMException,
    LLMConfigError,
    LLMAuthError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMAPIError,
)


class MockLLMClient:
    """Mock LLM client for deterministic unit testing without real network calls."""
    def __init__(self):
        self.call_count = 0

    def generate_response_sync(self, messages, system_instruction=None, temperature=0.7):
        self.call_count += 1
        return "• User shared their preference for Python and PostgreSQL.\n• Architecture requires microservices."

    async def generate_response(self, messages, system_instruction=None, temperature=0.7):
        return self.generate_response_sync(messages, system_instruction, temperature)


class TestContextManager(unittest.TestCase):

    def test_session_creation_and_message_addition(self):
        cm = ContextManager(strategy="sliding_window", max_history_messages=4)
        session = cm.get_or_create_session("test_session_1")
        self.assertEqual(session.session_id, "test_session_1")
        self.assertEqual(len(session.messages), 0)

        cm.add_message("test_session_1", "user", "Hello world")
        cm.add_message("test_session_1", "assistant", "Hi there! How can I help?")

        history = cm.get_history("test_session_1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello world")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Hi there! How can I help?")

    def test_sliding_window_under_capacity(self):
        cm = ContextManager(strategy="sliding_window", max_history_messages=6)
        cm.add_message("s1", "user", "Message 1")
        cm.add_message("s1", "assistant", "Response 1")

        payload, meta = cm.prepare_context_sync("s1")
        self.assertEqual(len(payload), 2)
        self.assertEqual(meta["strategy"], "sliding_window")
        self.assertEqual(meta["total_messages"], 2)
        self.assertEqual(meta["active_window_messages"], 2)
        self.assertEqual(meta["evicted_messages"], 0)
        self.assertEqual(payload[0]["content"], "Message 1")

    def test_sliding_window_truncation_over_capacity(self):
        # Set capacity to 4 messages (2 full turns)
        cm = ContextManager(strategy="sliding_window", max_history_messages=4)

        # Add 6 messages (3 turns)
        for i in range(1, 4):
            cm.add_message("s1", "user", f"User question {i}")
            cm.add_message("s1", "assistant", f"Assistant answer {i}")

        payload, meta = cm.prepare_context_sync("s1")

        # Should retain exactly the last 4 messages
        self.assertEqual(len(payload), 4)
        self.assertEqual(meta["strategy"], "sliding_window")
        self.assertEqual(meta["total_messages"], 6)
        self.assertEqual(meta["active_window_messages"], 4)
        self.assertEqual(meta["evicted_messages"], 2)

        # Oldest messages (User question 1, Assistant answer 1) must be truncated
        self.assertEqual(payload[0]["content"], "User question 2")
        self.assertEqual(payload[1]["content"], "Assistant answer 2")
        self.assertEqual(payload[2]["content"], "User question 3")
        self.assertEqual(payload[3]["content"], "Assistant answer 3")

    def test_running_summarization_strategy(self):
        mock_client = MockLLMClient()
        cm = ContextManager(strategy="summarization", max_history_messages=4, llm_client=mock_client)

        # Turn 1
        cm.add_message("s2", "user", "My favorite database is PostgreSQL and I love Rust.")
        cm.add_message("s2", "assistant", "PostgreSQL and Rust make an exceptional, robust combination.")

        # Turn 2
        cm.add_message("s2", "user", "I need to design an auth service.")
        cm.add_message("s2", "assistant", "Let's explore JWT vs session cookies.")

        # Under capacity (4 messages <= 4)
        payload, meta = cm.prepare_context_sync("s2")
        self.assertEqual(len(payload), 4)
        self.assertEqual(meta["evicted_messages"], 0)
        self.assertFalse(meta["has_summary"])

        # Turn 3: Exceeds capacity (now 6 messages > 4)
        cm.add_message("s2", "user", "What hashing algorithm should I use?")
        cm.add_message("s2", "assistant", "Argon2id is currently the recommended standard.")

        payload, meta = cm.prepare_context_sync("s2")

        # Evicted messages must be summarized!
        self.assertEqual(meta["strategy"], "summarization")
        self.assertEqual(meta["total_messages"], 6)
        self.assertEqual(meta["active_window_messages"], 4)
        self.assertEqual(meta["evicted_messages"], 2)
        self.assertTrue(meta["has_summary"])
        self.assertTrue("PostgreSQL" in meta["running_summary"] or "User" in meta["running_summary"])

        # In payload, memory context is injected before active window
        self.assertIn("[SYSTEM MEMORY CONTEXT", payload[0]["content"])
        self.assertEqual(payload[2]["content"], "I need to design an auth service.")
        self.assertEqual(payload[-1]["content"], "Argon2id is currently the recommended standard.")

    def test_session_clear(self):
        cm = ContextManager()
        cm.add_message("s3", "user", "Test message")
        self.assertEqual(len(cm.get_history("s3")), 1)

        cm.clear_session("s3")
        self.assertEqual(len(cm.get_history("s3")), 0)
        payload, meta = cm.prepare_context_sync("s3")
        self.assertEqual(len(payload), 0)
        self.assertEqual(meta["total_messages"], 0)

    def test_error_hierarchy(self):
        self.assertTrue(issubclass(LLMConfigError, LLMException))
        self.assertTrue(issubclass(LLMAuthError, LLMException))
        self.assertTrue(issubclass(LLMRateLimitError, LLMException))
        self.assertTrue(issubclass(LLMConnectionError, LLMException))
        self.assertTrue(issubclass(LLMAPIError, LLMException))


if __name__ == "__main__":
    unittest.main()
