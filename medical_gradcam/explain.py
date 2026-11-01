from pathlib import Path
import random

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


def _explain_image(model, device, transform, cam, image_path, classes, size, target=None):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(tensor).softmax(1)[0].cpu().tolist()
    predicted = int(np.argmax(probabilities))
    target_index = predicted if target is None else classes.index(target)
    heatmap = cam(input_tensor=tensor,
                  targets=[ClassifierOutputTarget(target_index)])[0]
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
    return resized, heatmap, overlay, predicted, probabilities, target_index


def explain(config, checkpoint, image_path, method="gradcam", target=None):
    device = select_device(config.device)
    model, classes, size, _ = load_model(checkpoint, config, device)
    cam_class = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}[method]
    with cam_class(model=model, target_layers=[model.layer4[-1]]) as cam:
        resized, heatmap, overlay, predicted, probabilities, target_index = _explain_image(
            model, device, image_transform(size), cam, image_path, classes, size, target)
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


def explain_gallery(config, checkpoint, predictions, n_images=20, selection="random",
                    method="gradcam", seed=None):
    """Display-sized pages of originals and attributions, loading weights only once."""
    if not isinstance(n_images, int) or n_images < 1:
        raise ValueError("n_images must be a positive integer.")
    if selection not in {"random", "correct", "incorrect"}:
        raise ValueError("selection must be random, correct or incorrect.")
    candidates = list(predictions)
    if selection != "random":
        candidates = [r for r in candidates
                      if (r["true"] == r["predicted"]) == (selection == "correct")]
    selected = random.Random(config.seed if seed is None else seed).sample(
        candidates, min(n_images, len(candidates)))
    result = {"requested_count": n_images, "available_count": len(candidates),
              "selection": selection, "seed": config.seed if seed is None else seed,
              "method": method, "checkpoint": str(Path(checkpoint).resolve()),
              "pages": [], "cases": []}
    if not selected:
        print(f"No cases available for selection='{selection}'. Check the filter and evaluation results.")
        return result
    device = select_device(config.device)
    model, classes, size, _ = load_model(checkpoint, config, device)
    transform = image_transform(size)
    cam_class = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}[method]
    output = Path(config.run_dir) / "explanations" / "gallery" / method / selection
    output.mkdir(parents=True, exist_ok=True)
    with cam_class(model=model, target_layers=[model.layer4[-1]]) as cam:
        for start in range(0, len(selected), 6):
            page_cases = selected[start:start + 6]
            rows = (len(page_cases) + 2) // 3
            fig, axes = plt.subplots(rows, 6, figsize=(18, rows * 3.6), squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            try:
                for index, case in enumerate(page_cases):
                    image_path = Path(config.data_dir) / case["path"]
                    original, _, overlay, pred, probs, _ = _explain_image(
                        model, device, transform, cam, image_path, classes, size)
                    left = axes[index // 3, (index % 3) * 2]
                    right = axes[index // 3, (index % 3) * 2 + 1]
                    correct = case["true"] == classes[pred]
                    left.imshow(original)
                    left.set_title(f"#{start + index + 1} | True: {case['true']}", fontsize=9)
                    right.imshow(overlay)
                    right.set_title(f"Pred: {classes[pred]} ({probs[pred]:.1%})\n"
                                    f"{'Correct' if correct else 'Incorrect'} | {method}",
                                    color="green" if correct else "crimson", fontsize=9)
                    result["cases"].append({"path": case["path"], "true": case["true"],
                        "predicted": classes[pred], "correct": correct,
                        "probabilities": dict(zip(classes, probs))})
                fig.suptitle(f"Original and predicted-class attribution | {selection} sample | "
                             f"{start + 1}-{start + len(page_cases)} of {len(selected)}")
                fig.tight_layout()
                page = output / f"page_{len(result['pages']) + 1:03d}.png"
                fig.savefig(page, dpi=120)
                result["pages"].append(str(page))
            finally:
                plt.close(fig)
    write_json(output / "selection.json", result)
    print(f"Showing {len(selected)} of {len(candidates)} available {selection} cases "
          f"(requested {n_images}). Saved {len(result['pages'])} gallery pages to {output}.")
    return result
