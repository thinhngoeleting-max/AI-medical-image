"""Controlled perturbations; no claim that altered images are clinically equivalent."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch

from .config import select_device, write_json
from .data import image_transform
from .evaluate import evaluation_data
from .model import load_model


def perturbations(image, border_fraction=0.08):
    array = np.array(image)
    height, width = array.shape[:2]
    border_h, border_w = max(1, round(height * border_fraction)), max(1, round(width * border_fraction))
    fill = np.median(array.reshape(-1, 3), axis=0).astype(np.uint8)
    border = array.copy()
    border[:border_h] = fill
    border[-border_h:] = fill
    border[:, :border_w] = fill
    border[:, -border_w:] = fill
    # Central occlusion has exactly the same number of pixels as the border mask.
    mask = np.zeros((height, width), dtype=bool)
    mask[:border_h] = mask[-border_h:] = True
    mask[:, :border_w] = mask[:, -border_w:] = True
    count = int(mask.sum())
    yy, xx = np.indices((height, width))
    distance = ((yy - (height - 1) / 2) / height) ** 2 + ((xx - (width - 1) / 2) / width) ** 2
    center = array.copy().reshape(-1, 3)
    center[np.argsort(distance, axis=None)[:count]] = fill
    return {
        "original": image,
        "border_mask": Image.fromarray(border),
        "center_mask_control": Image.fromarray(center.reshape(array.shape)),
        "brightness_0.9": ImageEnhance.Brightness(image).enhance(0.9),
        "brightness_1.1": ImageEnhance.Brightness(image).enhance(1.1),
        "translate_right_5pct": image.transform(image.size, Image.Transform.AFFINE,
            (1, 0, -round(width * 0.05), 0, 1, 0),
            resample=Image.Resampling.BILINEAR, fillcolor=tuple(int(v) for v in fill)),
    }


def robustness(config, checkpoint, split="val", limit=100, border_fraction=0.08):
    device = select_device(config.device)
    model, classes, size, metadata = load_model(checkpoint, config, device)
    records = evaluation_data(config, metadata, classes, split)
    rng = np.random.default_rng(config.seed)
    selected = sorted(rng.choice(len(records), min(limit, len(records)), replace=False).tolist())
    transform = image_transform(size)
    cases = []
    example_variants = None
    with torch.no_grad():
        for index in selected:
            record = records[index]
            with Image.open(Path(config.data_dir) / record["path"]) as source:
                image = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            variants = perturbations(image, border_fraction)
            if example_variants is None:
                example_variants = variants
            batch = torch.stack([transform(value) for value in variants.values()]).to(device)
            probabilities = model(batch).softmax(1).cpu().numpy()
            baseline_class = int(probabilities[0].argmax())
            for name, probs in zip(variants, probabilities):
                cases.append({"path": record["path"], "true": classes[record["label"]],
                    "variant": name, "prediction": classes[int(probs.argmax())],
                    "original_prediction": classes[baseline_class],
                    "prediction_changed": bool(probs.argmax() != baseline_class),
                    "original_class_probability_change": float(probs[baseline_class] - probabilities[0, baseline_class]),
                    "probabilities": dict(zip(classes, probs.tolist()))})
    summary = {}
    for name in example_variants:
        rows = [r for r in cases if r["variant"] == name]
        summary[name] = {
            "accuracy": sum(r["true"] == r["prediction"] for r in rows) / len(rows),
            "prediction_flip_rate": sum(r["prediction_changed"] for r in rows) / len(rows),
            "mean_original_class_probability_change": float(np.mean([
                r["original_class_probability_change"] for r in rows]))}
    output = Path(config.run_dir) / "robustness" / split
    write_json(output / "results.json", {"checkpoint": str(Path(checkpoint).resolve()),
        "split": split, "sample_count": len(selected), "total_available": len(records),
        "seed": config.seed, "border_fraction": border_fraction,
        "patient_separation_verified": metadata.get("patient_separation_verified", False),
        "sampling": "Seeded random sample without replacement",
        "limitations": "Perturbations can remove useful information and cause distribution shift. "
                       "Sensitivity alone does not prove shortcut learning. No lung segmentation is used.",
        "summary": summary, "cases": cases})
    for name, variant in example_variants.items():
        variant.save(output / f"example_{name}.png")
    print(f"Saved robustness results for {len(selected)} {split} images to {output}")
