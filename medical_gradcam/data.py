"""Persistent image splits; optional verified patient IDs, never filename guesses."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import write_json

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_transform(size, training=False):
    steps = [transforms.Resize((size, size))]
    if training:
        steps += [transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05))]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ImageRecords(Dataset):
    def __init__(self, root, records, classes, transform):
        self.root = Path(root)
        self.records = records
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record["label"]


def scan(root, split):
    folder = ImageFolder(Path(root) / split)
    records = [{"path": Path(p).relative_to(root).as_posix(), "label": int(y)}
               for p, y in folder.samples]
    return folder.classes, records


def fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def create_split(config, patient_csv=None):
    root = Path(config.data_dir)
    classes, records = scan(root, "train")
    test_classes, test = scan(root, "test")
    if classes != test_classes or classes != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Expected NORMAL and PNEUMONIA in both train and test.")
    indices = np.arange(len(records))
    labels = np.array([r["label"] for r in records])
    if patient_csv:
        with open(patient_csv, encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        patients = {}
        for row in rows:
            key = row["path"].replace("\\", "/")
            patient = row["patient_id"].strip()
            if not patient or (key in patients and patients[key] != patient):
                raise ValueError(f"Missing or conflicting patient ID: {key}")
            patients[key] = patient
        missing = [r["path"] for r in records + test if r["path"] not in patients]
        if missing:
            raise ValueError(f"Patient CSV must cover train and test. Missing: {missing[:3]}")
        groups = np.array([patients[r["path"]] for r in records])
        if set(groups) & {patients[r["path"]] for r in test}:
            raise ValueError("Verified patient overlap between training source and test set.")
        # Choose a reproducible group split close to the requested size and class mix.
        candidates = GroupShuffleSplit(n_splits=100, test_size=config.val_ratio,
                                       random_state=config.seed).split(indices, labels, groups)
        valid = [(a, b) for a, b in candidates
                 if len(set(labels[a])) == 2 and len(set(labels[b])) == 2]
        if not valid:
            raise ValueError("Cannot create patient-separated splits with both classes.")
        train_idx, val_idx = min(valid, key=lambda pair:
            abs(len(pair[1]) / len(records) - config.val_ratio)
            + abs(float(labels[pair[1]].mean() - labels.mean())))
        for r in records + test:
            r["patient_id"] = patients[r["path"]]
    else:
        train_idx, val_idx = train_test_split(indices, test_size=config.val_ratio,
            random_state=config.seed, stratify=labels)
    manifest = {
        "classes": classes, "seed": config.seed, "val_ratio": config.val_ratio,
        "patient_separation_verified": bool(patient_csv),
        "note": "Split from data/train; original data/val is unused. No images moved.",
        "train": [records[int(i)] for i in sorted(train_idx)],
        "val": [records[int(i)] for i in sorted(val_idx)], "test": test,
    }
    path = Path(config.run_dir) / "split.json"
    if path.exists():
        raise FileExistsError(f"{path} exists. Use it, or choose a new run_dir.")
    write_json(path, manifest)
    return manifest


def load_split(config):
    path = Path(config.run_dir) / "split.json"
    if not path.exists():
        raise FileNotFoundError("No split found. Run python -m medical_gradcam train to create it.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["seed"] != config.seed or manifest["val_ratio"] != config.val_ratio:
        raise ValueError("Split settings changed. Set a new run_dir in config.json and run train again.")
    return manifest
