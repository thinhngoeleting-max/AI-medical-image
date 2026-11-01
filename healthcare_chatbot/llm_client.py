"""Small Ollama /api/chat client using only the Python standard library."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaClient:
    def __init__(self, config):
        self.config = config
        self.opener = build_opener(ProxyHandler({}), _NoRedirects())

    def chat(self, messages):
        request = Request(self.config.base_url.rstrip("/") + "/api/chat",
            data=json.dumps({"model": self.config.model, "messages": messages,
                "stream": False, "think": False,
                "options": {"temperature": self.config.temperature,
                    "num_ctx": self.config.context_length,
                    "num_predict": self.config.max_output_tokens}}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.config.timeout_seconds) as response:
                data = json.load(response)
        except HTTPError as error:
            if error.code == 404:
                raise RuntimeError(f"Ollama could not find the endpoint or model. Check base_url and run: "
                                   f"ollama pull {self.config.model}") from error
            raise RuntimeError(f"Local Ollama request failed (HTTP {error.code}).") from error
        except (URLError, OSError) as error:
            raise RuntimeError("Cannot reach local Ollama. Start the Ollama app or run 'ollama serve', "
                               "then check base_url in chatbot_config.json.") from error
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("Local Ollama returned invalid JSON.") from error
        if not isinstance(data, dict):
            raise RuntimeError("Local Ollama returned an unexpected response.")
        if data.get("done_reason") == "length":
            raise RuntimeError("The answer was truncated. Increase max_output_tokens and retry.")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Local Ollama returned no answer. Check the model and server logs.")
        return content.strip()
