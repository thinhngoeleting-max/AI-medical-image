import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from medical_gradcam.config import Config, seed_everything
from medical_gradcam.data import create_split, fingerprint, image_transform, load_split
from medical_gradcam.evaluate import calibration_metrics, evaluate, save_ranking_curves
from medical_gradcam.explain import explain, explain_gallery
from medical_gradcam.model import load_model
from medical_gradcam.robustness import perturbations, robustness
from medical_gradcam.train import train


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        rng = np.random.default_rng(42)
        for split, count in [("train", 12), ("test", 4)]:
            for label in ("NORMAL", "PNEUMONIA"):
                directory = self.data / split / label
                directory.mkdir(parents=True)
                for index in range(count):
                    Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(
                        directory / f"{index}.png")
        self.config = Config(data_dir=str(self.data), run_dir=str(self.root / "run"),
            architecture="resnet18", val_ratio=0.25, image_size=32, batch_size=4,
            epochs=1, pretrained=False, device="cpu")

    def patient_csv(self, overlap=False):
        path = self.root / "patients.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "patient_id"])
            for image in sorted(self.data.rglob("*.png")):
                relative = image.relative_to(self.data)
                # Two views per patient; test identities use a different prefix.
                patient = f"{relative.parts[0]}_{relative.parts[1]}_{int(image.stem)//2}"
                if overlap and str(relative).startswith("test"):
                    patient = "train_NORMAL_0"
                writer.writerow([relative.as_posix(), patient])
        return path

    def test_image_split_is_disjoint_deterministic_and_nonmoving(self):
        original = set(self.data.rglob("*.png"))
        first = create_split(self.config)
        self.assertEqual([len(first[s]) for s in ("train", "val", "test")], [18, 6, 8])
        self.assertFalse(first["patient_separation_verified"])
        self.assertFalse({r["path"] for r in first["train"]} & {r["path"] for r in first["val"]})
        self.assertEqual(original, set(self.data.rglob("*.png")))
        self.config.run_dir = str(self.root / "repeat")
        self.assertEqual(fingerprint(first), fingerprint(create_split(self.config)))
        with self.assertRaises(FileExistsError):
            create_split(self.config)
        self.config.val_ratio = 0.3
        with self.assertRaises(ValueError):
            load_split(self.config)

    def test_verified_groups_and_overlap_rejection(self):
        manifest = create_split(self.config, self.patient_csv())
        self.assertTrue(manifest["patient_separation_verified"])
        groups = [{r["patient_id"] for r in manifest[s]} for s in ("train", "val", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.config.run_dir = str(self.root / "overlap")
        with self.assertRaisesRegex(ValueError, "overlap"):
            create_split(self.config, self.patient_csv(overlap=True))

    def test_calibration_and_perturbation_control(self):
        metrics = calibration_metrics([0, 1], [[1, 0], [0, 1]])
        self.assertEqual(metrics["brier_score"], 0)
        self.assertEqual(metrics["expected_calibration_error"], 0)
        image = Image.open(self.data / "test/NORMAL/0.png").convert("RGB")
        variants = perturbations(image)
        source = np.array(image)
        border_changes = np.any(np.array(variants["border_mask"]) != source, axis=2).sum()
        central_changes = np.any(np.array(variants["center_mask_control"]) != source, axis=2).sum()
        self.assertEqual(border_changes, central_changes)
        transform = image_transform(32)
        self.assertTrue(torch.equal(transform(image), transform(image)))

    def test_end_to_end_without_downloads(self):
        seed_everything(self.config.seed)
        train(self.config)
        manifest = load_split(self.config)
        self.assertEqual([len(manifest[s]) for s in ("train", "val", "test")], [18, 6, 8])
        checkpoint = Path(self.config.run_dir) / "best.pt"
        _, classes, size, metadata = load_model(checkpoint, self.config, torch.device("cpu"))
        self.assertEqual(metadata["split_fingerprint"], fingerprint(manifest))
        self.assertEqual(size, 32)
        self.assertEqual(classes, ["NORMAL", "PNEUMONIA"])
        metrics = evaluate(self.config, checkpoint)
        self.assertEqual(metrics["sample_count"], 8)
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 8)
        evaluation_dir = Path(self.config.run_dir) / "evaluation"
        curves = json.loads((evaluation_dir / "curves.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(curves["roc"]["auc"], metrics["roc_auc"])
        self.assertAlmostEqual(curves["precision_recall"]["average_precision"], metrics["average_precision"])
        self.assertTrue((evaluation_dir / "ranking_curves.png").exists())
        robustness(self.config, checkpoint, split="val", limit=3)
        image = self.data / "test/NORMAL/0.png"
        for method in ("gradcam", "gradcam++"):
            explain(self.config, checkpoint, image, method)
            output = Path(self.config.run_dir) / "explanations/0" / method
            self.assertTrue((output / "comparison.png").exists())
            heatmap = np.load(output / "heatmap.npy")
            self.assertEqual(heatmap.shape, (32, 32))
            self.assertTrue(np.isfinite(heatmap).all())
        predictions = json.loads((evaluation_dir / "predictions.json").read_text(encoding="utf-8"))
        gallery = explain_gallery(self.config, checkpoint, predictions, n_images=30)
        self.assertEqual(len(gallery["cases"]), 8)
        self.assertEqual(len({r["path"] for r in gallery["cases"]}), 8)
        self.assertEqual(len(gallery["pages"]), 2)
        self.assertTrue(all(Path(page).exists() for page in gallery["pages"]))
        incorrect = explain_gallery(self.config, checkpoint, predictions, n_images=2, selection="incorrect")
        self.assertEqual(len(incorrect["cases"]), 2)
        self.assertTrue(all(not r["correct"] for r in incorrect["cases"]))
        self.assertEqual(explain_gallery(self.config, checkpoint, [], n_images=20)["pages"], [])
        bad = dict(manifest)
        bad["train"] = manifest["train"][:-1]
        split_path = Path(self.config.run_dir) / "split.json"
        split_path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            evaluate(self.config, checkpoint)

    def test_training_reuses_patient_split_and_protects_previous_run(self):
        manifest = create_split(self.config, self.patient_csv())
        split_path = Path(self.config.run_dir) / "split.json"
        original = split_path.read_bytes()
        # A changed config must not silently reshuffle an existing split.
        self.config.val_ratio = 0.3
        with self.assertRaisesRegex(ValueError, "Split settings changed"):
            train(self.config)
        self.assertEqual(split_path.read_bytes(), original)

        self.config.val_ratio = 0.25
        train(self.config)
        self.assertEqual(split_path.read_bytes(), original)
        checkpoint = Path(self.config.run_dir) / "best.pt"
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(metadata["split_fingerprint"], fingerprint(manifest))
        self.assertTrue(metadata["patient_separation_verified"])
        # Another train invocation must leave both split and checkpoint intact.
        previous_mtime = checkpoint.stat().st_mtime_ns
        with self.assertRaisesRegex(FileExistsError, "best.pt already exists"):
            train(self.config)
        self.assertEqual(checkpoint.stat().st_mtime_ns, previous_mtime)
        self.assertEqual(split_path.read_bytes(), original)

    def test_ranking_curves_use_probabilities_and_handle_one_class(self):
        output = self.root / "curves"
        perfect = save_ranking_curves([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], output)
        self.assertEqual(perfect["roc"]["auc"], 1.0)
        self.assertEqual(perfect["precision_recall"]["average_precision"], 1.0)
        reversed_scores = save_ranking_curves([0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2], output)
        self.assertEqual(reversed_scores["roc"]["auc"], 0.0)
        unavailable = save_ranking_curves([0, 0], [0.1, 0.2], output)
        self.assertIsNone(unavailable["roc"])
        self.assertIsNone(unavailable["precision_recall"])
        self.assertIn("Both ground-truth classes are required", (output / "curves.json").read_text())


if __name__ == "__main__":
    unittest.main()
