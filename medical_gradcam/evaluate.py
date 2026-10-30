from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, average_precision_score)
from torch.utils.data import DataLoader

from .config import select_device, write_json
from .data import ImageRecords, fingerprint, image_transform, load_split, scan
from .model import load_model


def calibration_metrics(labels, probabilities, bins=10):
    """Positive-class Brier score and top-label expected calibration error."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    for index in range(bins):
        mask = (confidence >= index / bins) & (
            confidence <= 1 if index == bins - 1 else confidence < (index + 1) / bins)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {"brier_score": float(np.mean((probabilities[:, 1] - labels) ** 2)),
            "expected_calibration_error": ece, "calibration_bins": bins}


def evaluation_data(config, metadata, classes, split):
    if metadata.get("legacy") and split == "test":
        found_classes, records = scan(config.data_dir, "test")
        if found_classes != classes:
            raise ValueError("Test classes disagree with checkpoint classes.")
        return records
    if metadata.get("legacy"):
        raise ValueError("Legacy checkpoint has no split provenance; validation evaluation is unsupported.")
    manifest = load_split(config)
    if fingerprint(manifest) != metadata["split_fingerprint"]:
        raise ValueError("Checkpoint and split manifest disagree.")
    if manifest["classes"] != classes:
        raise ValueError("Split and checkpoint classes disagree.")
    return manifest[split]


def evaluate(config, checkpoint, split="test", limit=None):
    device = select_device(config.device)
    model, classes, size, metadata = load_model(checkpoint, config, device)
    records = evaluation_data(config, metadata, classes, split)
    total_available = len(records)
    if limit is not None:
        records = records[:limit]
    dataset = ImageRecords(config.data_dir, records, classes, image_transform(size))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers)
    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            probabilities.extend(model(images.to(device)).softmax(1).cpu().tolist())
    truth = [r["label"] for r in records]
    predicted = np.argmax(probabilities, axis=1).tolist()
    report = classification_report(truth, predicted, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(truth, predicted, labels=list(range(len(classes))))
    positive = np.array(probabilities)[:, 1]
    metrics = {"checkpoint": str(Path(checkpoint).resolve()), "split": split,
        "sample_count": len(records), "total_available": total_available,
        "partial_evaluation": len(records) < total_available,
        "patient_separation_verified": metadata.get("patient_separation_verified", False),
        "classes": classes, "accuracy": float(np.mean(np.array(truth) == predicted)),
        "classification_report": report, "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc_score(truth, positive)) if len(set(truth)) == 2 else None,
        "average_precision": float(average_precision_score(truth, positive))
            if len(set(truth)) == 2 else None,
        **calibration_metrics(truth, probabilities)}
    results = []
    for record, pred, probs in zip(records, predicted, probabilities):
        true = record["label"]
        category = ("true_positive" if true == 1 else "true_negative") if true == pred else (
            "false_positive" if pred == 1 else "false_negative")
        results.append({"path": record["path"], "true": classes[true],
            "predicted": classes[pred], "category": category,
            "probabilities": dict(zip(classes, probs))})
    output = Path(config.run_dir) / ("evaluation" if split == "test" else "validation")
    if limit is not None:
        output = output / f"sample_{limit}"
    write_json(output / "metrics.json", metrics)
    write_json(output / "predictions.json", results)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot()
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close("all")
    print(f"{split}: {len(records)}/{total_available} images | accuracy {metrics['accuracy']:.4f}")
    print(f"Confusion matrix (rows=true, columns=predicted): {cm.tolist()}")
    print(f"Saved metrics and predictions to {output}")
    return metrics
