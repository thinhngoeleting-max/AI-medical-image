from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import seed_everything, select_device, write_json
from .data import ImageRecords, create_split, fingerprint, image_transform, load_split
from .model import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return {"loss": total_loss / total, "accuracy": correct / total}


def train(config):
    seed_everything(config.seed)
    output = Path(config.run_dir)
    if (output / "best.pt").exists():
        raise FileExistsError("best.pt already exists. Choose a new run_dir for another experiment.")
    if (output / "split.json").exists():
        manifest = load_split(config)
        print(f"Reusing saved split: {output / 'split.json'}", flush=True)
    else:
        print(f"Creating split from config: validation ratio {config.val_ratio:.0%}", flush=True)
        manifest = create_split(config)
        print(f"Split saved to {output / 'split.json'}. No images moved.", flush=True)
    print(" | ".join(f"{name}: {len(manifest[name])}" for name in ("train", "val", "test")),
          flush=True)
    device = select_device(config.device)
    print(f"Training {config.architecture} on {device}", flush=True)
    if not manifest["patient_separation_verified"]:
        print("Patient separation is UNVERIFIED: this is an image-stratified split.", flush=True)
    loaders = {}
    for split in ("train", "val"):
        dataset = ImageRecords(config.data_dir, manifest[split], manifest["classes"],
                               image_transform(config.image_size, training=split == "train"))
        loaders[split] = DataLoader(dataset, batch_size=config.batch_size,
            shuffle=split == "train", num_workers=config.num_workers,
            pin_memory=device.type == "cuda")
    model = build_model(config.architecture, len(manifest["classes"]), config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history, best = [], (-1.0, float("-inf"))
    write_json(output / "config.json", config.to_dict())
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        score = (val_metrics["accuracy"], -val_metrics["loss"])
        if score > best:
            best = score
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                "config": config.to_dict(), "classes": manifest["classes"],
                "split_fingerprint": fingerprint(manifest),
                "patient_separation_verified": manifest["patient_separation_verified"],
                "validation": val_metrics}, output / "best.pt")
        write_json(output / "history.json", history)
        print(f"Epoch {epoch}/{config.epochs} | train loss {train_metrics['loss']:.4f} "
              f"acc {train_metrics['accuracy']:.4f} | val loss {val_metrics['loss']:.4f} "
              f"acc {val_metrics['accuracy']:.4f}", flush=True)
    print(f"Saved best checkpoint: {output / 'best.pt'}", flush=True)
