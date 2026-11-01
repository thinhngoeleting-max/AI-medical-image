import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class ChatbotConfig:
    model: str = "gemma4:e2b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    context_length: int = 8192
    max_output_tokens: int = 1200
    timeout_seconds: int = 180
    max_report_chars: int = 12000
    max_question_chars: int = 2000
    max_history_turns: int = 6

    def __post_init__(self):
        parsed = urlparse(self.base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError("base_url must be a local HTTP Ollama address, e.g. http://127.0.0.1:11434")
        if not self.model.strip() or "cloud" in self.model.lower():
            raise ValueError("Choose a downloaded local Ollama model, not a cloud model.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        for name in ("context_length", "max_output_tokens", "timeout_seconds", "max_report_chars",
                     "max_question_chars", "max_history_turns"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def load(cls, path="chatbot_config.json"):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
