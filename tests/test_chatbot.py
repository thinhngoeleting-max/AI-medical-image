import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, Mock

from healthcare_chatbot.config import ChatbotConfig
from healthcare_chatbot.explainer import ReportExplainer
from healthcare_chatbot.llm_client import OllamaClient


class ChatbotTests(unittest.TestCase):
    def test_config_rejects_remote_and_cloud_targets(self):
        for address in ("https://example.com", "http://127.0.0.1.evil.example", "http://user@localhost:11434"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                ChatbotConfig(base_url=address)
        with self.assertRaises(ValueError):
            ChatbotConfig(model="qwen3:cloud")
        with self.assertRaises(ValueError):
            ChatbotConfig(max_history_turns=0)

    def test_report_is_preserved_and_followups_are_bounded(self):
        client = Mock()
        client.chat.return_value = "Example response, not a validated medical explanation."
        source = "No focal opacity. An infective process cannot be excluded."
        config = ChatbotConfig(max_history_turns=1)
        session = ReportExplainer(source, config, client)
        session.ask("Explain the wording.")
        session.ask("Does this confirm a diagnosis?")
        messages = session.messages_for("What information is missing?")
        self.assertEqual(json.loads(messages[1]["content"])["report_text"], source)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], "Does this confirm a diagnosis?")
        self.assertNotIn("Explain the wording.", [m["content"] for m in messages])
        client.chat.side_effect = RuntimeError("Server unavailable")
        previous = list(session.history)
        with self.assertRaises(RuntimeError):
            session.ask("Try again.")
        self.assertEqual(session.history, previous)

    def test_input_limits_do_not_silently_clip_source(self):
        config = ChatbotConfig(max_report_chars=10, max_question_chars=5)
        with self.assertRaises(ValueError):
            ReportExplainer(" ", config)
        with self.assertRaises(ValueError):
            ReportExplainer("Long report text", config)
        session = ReportExplainer("No change.", config, Mock())
        with self.assertRaises(ValueError):
            session.ask("An excessively long question")

    def test_ollama_request_and_incomplete_response_handling(self):
        client = OllamaClient(ChatbotConfig())
        client.opener = MagicMock()
        response = io.BytesIO(json.dumps({"message": {"content": "Explanation"}, "done": True}).encode())
        client.opener.open.return_value.__enter__.return_value = response
        messages = [{"role": "user", "content": "Fictional report"}]
        self.assertEqual(client.chat(messages), "Explanation")
        request = client.opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["messages"], messages)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        response = io.BytesIO(json.dumps({"message": {"content": "Partial"}, "done_reason": "length"}).encode())
        client.opener.open.return_value.__enter__.return_value = response
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            client.chat(messages)

    def test_dry_run_cli_needs_no_server(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-m", "healthcare_chatbot", "--example", "negation", "--dry-run"],
                                cwd=root, capture_output=True, text=True, check=True, timeout=15)
        messages = json.loads(result.stdout)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("FICTIONAL EDUCATIONAL EXAMPLE", json.loads(messages[1]["content"])["report_text"])


if __name__ == "__main__":
    unittest.main()
