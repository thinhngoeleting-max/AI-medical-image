from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
import random
from threading import Lock
import time
from urllib.request import Request
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request as WebRequest
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from healthcare_chatbot.config import ChatbotConfig
from healthcare_chatbot.image_assistant import ImageResultAssistant
from healthcare_chatbot.llm_client import OllamaClient
from medical_gradcam.config import Config
from medical_gradcam.inference import ImageAnalyzer, MAX_UPLOAD_BYTES

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"


def available_models(config):
    try:
        client = OllamaClient(config)
        with client.opener.open(Request(config.base_url.rstrip("/") + "/api/tags"), timeout=2) as response:
            data = json.load(response)
        names = [entry["name"] for entry in data.get("models", [])
                 if isinstance(entry.get("name"), str) and "cloud" not in entry["name"].lower()]
        return {"online": True, "models": names}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"online": False, "models": []}


class Sessions:
    """Bounded in-memory contexts. Images and conversations are not written to disk."""
    def __init__(self):
        self.items = OrderedDict()
        self.lock = Lock()

    def add(self, assistant):
        key = uuid4().hex
        with self.lock:
            self.items[key] = (time.monotonic(), assistant, Lock())
            while len(self.items) > 20:
                self.items.popitem(last=False)
        return key

    def get(self, key):
        with self.lock:
            entry = self.items.get(key)
            if entry is None or time.monotonic() - entry[0] > 1800:
                self.items.pop(key, None)
                raise HTTPException(404, "This analysis expired. Analyse the image again to continue.")
            return entry[1], entry[2]


class ChatRequest(BaseModel):
    analysis_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(default=None, max_length=120)


def create_app(config_path="config.json", chat_config_path="chatbot_config.json", analyzer=None):
    config_path, chat_config_path = Path(config_path), Path(chat_config_path)
    config = Config.load(config_path if config_path.is_absolute() else ROOT / config_path)
    config = replace(config, data_dir=str(ROOT / config.data_dir), run_dir=str(ROOT / config.run_dir))
    chat_config = ChatbotConfig.load(chat_config_path if chat_config_path.is_absolute() else ROOT / chat_config_path)
    checkpoint = Path(config.run_dir) / "best.pt"
    analyzer = analyzer or ImageAnalyzer(config, checkpoint)
    sessions = Sessions()
    app = FastAPI(title="Clarity local X-ray workspace", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    examples = []
    rng = random.Random(config.seed)
    for label in ("NORMAL", "PNEUMONIA"):
        files = sorted(p for p in (Path(config.data_dir) / "test" / label).glob("*")
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for path in rng.sample(files, min(4, len(files))):
            examples.append({"id": len(examples), "label": label, "path": path})

    @app.middleware("http")
    async def local_requests(request, call_next):
        # Prevent unrelated web pages from posting to this unauthenticated local app.
        origin = request.headers.get("origin")
        own_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if origin and origin != own_origin:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'")
        return response

    @app.get("/api/status")
    def status():
        return {"checkpoint_ready": checkpoint.exists(), "architecture": config.architecture,
                "configured_model": chat_config.model, "ollama": available_models(chat_config)}

    @app.get("/api/examples")
    def list_examples():
        return [{"id": e["id"], "label": e["label"], "image_url": f"/api/examples/{e['id']}/image"}
                for e in examples]

    def get_example(example_id):
        if example_id < 0 or example_id >= len(examples):
            raise HTTPException(404, "Example image not found.")
        return examples[example_id]

    @app.get("/api/examples/{example_id}/image")
    def example_image(example_id: int):
        return FileResponse(get_example(example_id)["path"])

    def perform_analysis(content, source_label=None):
        try:
            result = analyzer.analyze(content)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, "Image analysis failed. Check the checkpoint, available memory, "
                                "and server terminal; finish training before using the interface.") from error
        result["analysis_id"] = sessions.add(ImageResultAssistant(result, chat_config))
        result["dataset_label"] = source_label
        return result

    @app.post("/api/analyze")
    async def analyze(request: WebRequest):
        # Bounded raw upload stays in memory, avoiding multipart temporary files.
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Choose a JPEG or PNG image up to 10 MB.")
            content.extend(chunk)
        return await run_in_threadpool(perform_analysis, bytes(content))

    @app.post("/api/examples/{example_id}/analyze")
    def analyze_example(example_id: int):
        example = get_example(example_id)
        return perform_analysis(example["path"].read_bytes(), example["label"])

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        assistant, lock = sessions.get(request.analysis_id)
        model = request.model or chat_config.model
        models = available_models(chat_config)
        if not models["online"]:
            raise HTTPException(503, "Ollama is offline. Start the Ollama app or run ollama serve.")
        if model not in models["models"]:
            raise HTTPException(400, "Select an installed local model. Download it with ollama pull first.")
        try:
            with lock:
                answer = assistant.ask(request.question, model)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        return {"answer": answer, "model": model}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
