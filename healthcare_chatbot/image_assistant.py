"""Discuss measured classifier output; never turn it into a radiology report."""
import json
from pathlib import Path

from .llm_client import OllamaClient


class ImageResultAssistant:
    def __init__(self, analysis, config):
        self.config = config
        self.context = {key: analysis[key] for key in
                        ("prediction", "model_probability", "probabilities", "model", "limitations")}
        self.prompt = (Path(__file__).parent / "prompts" / "image_result.txt").read_text(encoding="utf-8")
        self.history = []

    def ask(self, question, model=None):
        from dataclasses import replace
        if not isinstance(question, str) or not question.strip() or len(question) > self.config.max_question_chars:
            raise ValueError(f"Ask a question between 1 and {self.config.max_question_chars} characters.")
        config = replace(self.config, model=model) if model else self.config
        messages = [{"role": "system", "content": self.prompt},
                    {"role": "user", "content": json.dumps({"classifier_measurements": self.context})},
                    *self.history, {"role": "user", "content": question.strip()}]
        answer = OllamaClient(config).chat(messages)
        self.history.extend([messages[-1], {"role": "assistant", "content": answer}])
        self.history = self.history[-2 * self.config.max_history_turns:]
        return answer
