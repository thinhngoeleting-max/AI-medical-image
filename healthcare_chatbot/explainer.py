import json
from pathlib import Path

from .llm_client import OllamaClient


class ReportExplainer:
    def __init__(self, report_text, config, client=None):
        if not isinstance(report_text, str) or not report_text.strip():
            raise ValueError("Provide a nonempty report as text.")
        if len(report_text) > config.max_report_chars:
            raise ValueError("Report exceeds max_report_chars. Use a shorter report; text is not silently truncated.")
        self.config = config
        self.client = client if client is not None else OllamaClient(config)
        self.report_text = report_text.strip()
        self.history = []
        self.system_prompt = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")

    def messages_for(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Ask a nonempty question.")
        if len(question) > self.config.max_question_chars:
            raise ValueError("Question exceeds max_question_chars.")
        # Always retain the source report; bound follow-up history without clipping it.
        history = self.history[-2 * self.config.max_history_turns:]
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps({"report_text": self.report_text}, ensure_ascii=False)},
            *[dict(message) for message in history],
            {"role": "user", "content": question.strip()},
        ]

    def ask(self, question="Please explain this report in plain language."):
        messages = self.messages_for(question)
        answer = self.client.chat(messages)
        # Commit the turn only after a successful response. Nothing is written to disk.
        self.history.extend([messages[-1], {"role": "assistant", "content": answer}])
        self.history = self.history[-2 * self.config.max_history_turns:]
        return answer
