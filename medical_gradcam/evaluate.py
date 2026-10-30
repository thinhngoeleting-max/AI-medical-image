from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
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


def save_ranking_curves(labels, scores, output, positive_class="PNEUMONIA"):
    """Save ranking curves from probabilities, not thresholded class predictions."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    curves = {"positive_class": positive_class, "sample_count": len(labels),
              "positive_prevalence": float(labels.mean()), "roc": None,
              "precision_recall": None}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        curves["roc"] = {"false_positive_rate": fpr.tolist(),
                         "true_positive_rate": tpr.tolist(), "auc": auc}
        curves["precision_recall"] = {"precision": precision.tolist(),
                                      "recall": recall.tolist(), "average_precision": ap}
        axes[0].plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
        axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
        axes[1].step(recall, precision, where="post", label=f"{positive_class} AP = {ap:.3f}")
        axes[1].axhline(curves["positive_prevalence"], linestyle="--", color="gray",
                        label=f"Prevalence = {curves['positive_prevalence']:.3f}")
        for ax in axes:
            ax.legend(loc="lower left")
    else:
        curves["note"] = "Both ground-truth classes are required; ranking curves are unavailable."
        for ax in axes:
            ax.text(0.5, 0.5, "Unavailable: evaluation contains only one class",
                    ha="center", va="center", wrap=True, transform=ax.transAxes)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate (recall)")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.05))
        ax.grid(alpha=0.2)
    fig.suptitle(f"Positive class: {positive_class} | n={len(labels)}")
    fig.tight_layout()
    fig.savefig(output / "ranking_curves.png", dpi=140)
    plt.close(fig)
    write_json(output / "curves.json", curves)
    return curves


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
        "classes": classes, "positive_class": classes[1],
        "accuracy": float(np.mean(np.array(truth) == predicted)),
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
    plt.close(disp.figure_)
    save_ranking_curves(truth, positive, output, classes[1])
    print(f"{split}: {len(records)}/{total_available} images | accuracy {metrics['accuracy']:.4f}")
    print(f"Confusion matrix (rows=true, columns=predicted): {cm.tolist()}")
    print(f"Saved metrics and predictions to {output}")
    return metrics
