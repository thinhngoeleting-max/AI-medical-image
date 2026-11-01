"""In-memory image inference for the local UI. Uploaded images are never saved."""
import base64
from io import BytesIO
from pathlib import Path
from threading import Lock
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import select_device
from .data import image_transform
from .model import load_model

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def png_data_url(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(content):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a JPEG or PNG image up to 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported. Convert DICOM files separately.")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Image is too large. Use an image with at most 20 million pixels.")
                return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Could not read this image. Choose a valid JPEG or PNG.") from error


class ImageAnalyzer:
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.lock = Lock()
        self.model = None
        self.signature = None

    def analyze(self, content):
        image = decode_image(content)
        with self.lock:
            if not self.checkpoint.exists():
                raise FileNotFoundError("No trained checkpoint found. Run python -m medical_gradcam train first.")
            signature = (self.checkpoint.stat().st_mtime_ns, self.checkpoint.stat().st_size)
            if self.model is None or signature != self.signature:
                self.device = select_device(self.config.device)
                self.model, self.classes, self.size, self.metadata = load_model(
                    self.checkpoint, self.config, self.device)
                if self.classes != ["NORMAL", "PNEUMONIA"]:
                    raise ValueError("This interface expects NORMAL/PNEUMONIA checkpoint classes.")
                self.transform = image_transform(self.size)
                self.signature = signature
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probabilities = self.model(tensor).softmax(1)[0].cpu().tolist()
            predicted = int(np.argmax(probabilities))
            with torch.enable_grad(), GradCAM(model=self.model, target_layers=[self.model.layer4[-1]]) as cam:
                heatmap = cam(input_tensor=tensor,
                              targets=[ClassifierOutputTarget(predicted)])[0]
            self.model.zero_grad(set_to_none=True)
            resized = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            rgb = np.asarray(resized, dtype=np.float32) / 255.0
            overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
            return {"prediction": self.classes[predicted],
                "model_probability": probabilities[predicted],
                "probabilities": dict(zip(self.classes, probabilities)),
                "original_image": png_data_url(resized),
                "gradcam_image": png_data_url(Image.fromarray(overlay)),
                "model": {"architecture": self.metadata.get("config", {}).get(
                    "architecture", self.config.architecture), "epoch": self.metadata.get("epoch"),
                    "image_size": self.size, "device": str(self.device),
                    "patient_separation_verified": self.metadata.get("patient_separation_verified", False)},
                "limitations": "Experimental NORMAL/PNEUMONIA classifier. Model probability is not "
                    "a clinical certainty. This model cannot verify image type, diagnose illness, "
                    "or rule out other conditions. Grad-CAM shows attribution, not lesion boundaries."}
