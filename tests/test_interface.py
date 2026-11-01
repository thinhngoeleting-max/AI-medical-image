from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import torch

from app.server import create_app
from healthcare_chatbot.config import ChatbotConfig
from healthcare_chatbot.image_assistant import ImageResultAssistant
from medical_gradcam.config import Config
from medical_gradcam.inference import MAX_UPLOAD_BYTES, decode_image
from medical_gradcam.model import build_model


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        data = cls.root / "data"
        pixels = np.random.default_rng(42).integers(0, 256, (48, 40, 3), dtype=np.uint8)
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        cls.image_bytes = buffer.getvalue()
        for label in ("NORMAL", "PNEUMONIA"):
            directory = data / "test" / label
            directory.mkdir(parents=True)
            (directory / "example.png").write_bytes(cls.image_bytes)
        run = cls.root / "run"
        run.mkdir()
        config = Config(data_dir=str(data), run_dir=str(run), architecture="resnet18",
                        image_size=32, pretrained=False, device="cpu")
        cls.checkpoint = run / "best.pt"
        torch.save({"model_state": build_model("resnet18").state_dict(), "config": config.to_dict(),
                    "classes": ["NORMAL", "PNEUMONIA"], "epoch": 1,
                    "patient_separation_verified": False}, cls.checkpoint)
        cls.config_path = cls.root / "config.json"
        cls.config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        cls.chat_path = cls.root / "chat.json"
        cls.chat_path.write_text(json.dumps(asdict(ChatbotConfig())), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.client = TestClient(create_app(self.config_path, self.chat_path), base_url="http://localhost")
        self.addCleanup(self.client.close)

    def test_upload_and_example_inference_match_without_writing_outputs(self):
        before = set(self.root.rglob("*"))
        checkpoint_time = self.checkpoint.stat().st_mtime_ns
        upload = self.client.post("/api/analyze", content=self.image_bytes,
                                  headers={"Content-Type": "image/png"})
        self.assertEqual(upload.status_code, 200, upload.text)
        result = upload.json()
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertIsNone(result["dataset_label"])
        self.assertEqual(result["model"]["architecture"], "resnet18")
        for key in ("original_image", "gradcam_image"):
            self.assertTrue(result[key].startswith("data:image/png;base64,"))
        example = self.client.post("/api/examples/0/analyze")
        self.assertEqual(example.status_code, 200, example.text)
        self.assertEqual(example.json()["dataset_label"], "NORMAL")
        self.assertEqual(result["probabilities"], example.json()["probabilities"])
        self.assertEqual(before, set(self.root.rglob("*")))
        self.assertEqual(self.checkpoint.stat().st_mtime_ns, checkpoint_time)

    def test_bad_upload_missing_example_and_origin_are_handled(self):
        self.assertEqual(self.client.post("/api/analyze", content=b"not an image").status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", content=b"").status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", content=b"0" * (MAX_UPLOAD_BYTES + 1)).status_code, 413)
        self.assertEqual(self.client.get("/api/examples/999/image").status_code, 404)
        self.assertEqual(self.client.post("/api/analyze", content=self.image_bytes,
                                         headers={"Origin": "https://unrelated.example"}).status_code, 403)
        gif = BytesIO()
        Image.new("RGB", (10, 10)).save(gif, format="GIF")
        with self.assertRaisesRegex(ValueError, "Only JPEG and PNG"):
            decode_image(gif.getvalue())

    def test_ui_status_and_offline_chat(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        result = self.client.post("/api/analyze", content=self.image_bytes).json()
        with patch("app.server.available_models", return_value={"online": False, "models": []}):
            status = self.client.get("/api/status").json()
            self.assertTrue(status["checkpoint_ready"])
            self.assertFalse(status["ollama"]["online"])
            response = self.client.post("/api/chat", json={"analysis_id": result["analysis_id"],
                                                          "question": "Explain this score."})
            self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.post("/api/chat", json={"analysis_id": "unknown", "question": "Explain."}).status_code, 404)

    def test_chat_uses_server_measurements_and_selected_local_model(self):
        result = self.client.post("/api/analyze", content=self.image_bytes).json()
        model = "gemma4:e2b"
        with patch("app.server.available_models", return_value={"online": True, "models": [model]}), \
                patch("healthcare_chatbot.image_assistant.OllamaClient") as client_class:
            client_class.return_value.chat.return_value = "This is an experimental model score, not a diagnosis."
            response = self.client.post("/api/chat", json={"analysis_id": result["analysis_id"],
                "question": "What does this prediction mean?", "model": model})
            self.assertEqual(response.status_code, 200, response.text)
            messages = client_class.return_value.chat.call_args.args[0]
            context = json.loads(messages[1]["content"])["classifier_measurements"]
            self.assertEqual(context["probabilities"], result["probabilities"])
            self.assertNotIn("original_image", context)
            self.assertNotIn("gradcam_image", context)
            self.assertEqual(client_class.call_args.args[0].model, model)
            self.assertEqual(response.json()["model"], model)


if __name__ == "__main__":
    unittest.main()
