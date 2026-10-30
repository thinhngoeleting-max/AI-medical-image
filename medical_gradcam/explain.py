from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import select_device, write_json
from .data import image_transform
from .model import load_model


def explain(config, checkpoint, image_path, method="gradcam", target=None):
    device = select_device(config.device)
    model, classes, size, _ = load_model(checkpoint, config, device)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = image_transform(size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(tensor).softmax(1)[0].cpu().tolist()
    predicted = int(np.argmax(probabilities))
    target_index = predicted if target is None else classes.index(target)
    cam_class = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}[method]
    with cam_class(model=model, target_layers=[model.layer4[-1]]) as cam:
        heatmap = cam(input_tensor=tensor,
                      targets=[ClassifierOutputTarget(target_index)])[0]
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
    output = Path(config.run_dir) / "explanations" / Path(image_path).stem / method
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output / "overlay.png")
    np.save(output / "heatmap.npy", heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, visual, title in zip(axes, [resized, heatmap, overlay],
            ["Resized input", f"{method}: {classes[target_index]}",
             f"Prediction: {classes[predicted]} ({probabilities[predicted]:.1%})"]):
        ax.imshow(visual, cmap="jet" if title.startswith(method) else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)
    write_json(output / "prediction.json", {"image": str(image_path),
        "checkpoint": str(Path(checkpoint).resolve()), "method": method,
        "prediction": classes[predicted], "target_class": classes[target_index],
        "probabilities": dict(zip(classes, probabilities)),
        "interpretation": "Model attribution, not a lesion annotation or diagnosis."})
    print(f"Saved heatmap, overlay and comparison to {output}")
