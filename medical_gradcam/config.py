import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Config:
    data_dir: str = "data"
    run_dir: str = "runs/baseline"
    architecture: str = "resnet50"
    val_ratio: float = 0.2
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    pretrained: bool = True

    @classmethod
    def load(cls, path):
        config = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        if not 0 < config.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1.")
        if min(config.image_size, config.batch_size, config.epochs) < 1:
            raise ValueError("image_size, batch_size and epochs must be positive.")
        if config.num_workers < 0 or config.learning_rate <= 0:
            raise ValueError("num_workers must be nonnegative and learning_rate positive.")
        return config

    def to_dict(self):
        return asdict(self)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
