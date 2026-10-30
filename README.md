# Chest X-ray explainability experiments

Python modules for ResNet classification, Grad-CAM, and controlled image perturbations.
Run training from this project directory in the terminal. Open `Evaluation.ipynb`
to display evaluation, Grad-CAM and robustness results in Jupyter.
This is an educational research project.

## Setup

Your existing `.venv` already contains the main dependencies. In PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m medical_gradcam --help
```

If activation is unavailable, use `.\.venv\Scripts\python.exe` instead of `python`.
For missing packages, run `python -m pip install -r requirements.txt`. Keep the
working CUDA-enabled PyTorch installation; reinstalling it is unnecessary.

## Settings and data

Edit `config.json`. Set `val_ratio` to `0.20` for approximately 80% training / 20%
validation **from `data/train`**. The original `data/val` is unused. `data/test`
stays separate. Files are never moved or copied by the split command.

Expected folders:

```text
data/
  train/NORMAL/
  train/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
```

Paths in the config are relative to the terminal's working directory. New runs go
under `runs/baseline`. Training creates this folder and its split automatically.
Set a different `run_dir` for each new configuration or model comparison.
Supported architectures: `resnet18`, `resnet50`. `pretrained: true` downloads
ImageNet weights if they are not already cached. Seeds are fixed, although exact
results can vary across devices and library versions. Training uses small affine
augmentation; evaluation uses deterministic resizing and ImageNet normalization.

## Train and evaluate

```powershell
python -m medical_gradcam train
python -m medical_gradcam evaluate
python -m medical_gradcam robustness --split val --limit 100
```

Only `train` is needed to start: it creates the split using `val_ratio` and `seed`
from `config.json` if none exists, or reuses the existing matching split. It prints
the split sizes and training progress in the terminal. The separate `split`
command is optional, for inspecting a split or providing verified patient IDs.

The split is persisted in `runs/baseline/split.json`. Reusing it preserves the
same membership. If changing the validation ratio or seed, choose a new `run_dir`
in `config.json` and run `train` again. Existing split manifests and trained
checkpoints are protected against accidental replacement.
Checkpoint selection maximizes validation accuracy, then minimizes validation
loss to break ties. Checkpoints include preprocessing settings, class names,
validation metrics and the split fingerprint. Training is not resumed by rerunning
the command. The fingerprint checks manifest membership, not image file contents.

Use validation for choosing settings. Reserve test evaluation for the final
comparison. Evaluation exports probabilities, per-class precision/recall/F1,
confusion matrix, ROC-AUC, average precision, Brier score and a 10-bin calibration
error. Calibration metrics measure reliability; they do not calibrate the model.
`predictions.json` categorizes true/false positives and negatives, so you can choose
examples from each group for explanations. A `--limit` evaluation is explicitly
marked partial and saved separately from full metrics.

### Patient separation

The default split is stratified **by image**, not verified patient identity.
Existing filenames are insufficient to establish identity: pneumonia filename
prefixes recur across train and test and need source documentation to interpret.
For a stronger experiment, obtain verified IDs and create a CSV covering every
image in `data/train` and `data/test`:

```csv
path,patient_id
train/NORMAL/example.jpeg,verified_patient_a
train/PNEUMONIA/example.jpeg,verified_patient_b
test/NORMAL/example.jpeg,verified_patient_c
```

With a fresh `run_dir`, run:

```powershell
python -m medical_gradcam split --patient-csv patients.csv
```

This keeps patients together and rejects verified train/test patient overlap.
The chosen group split approximates the requested validation size and class mix;
exact proportions depend on patient group sizes. The CSV example is a schema,
not actual IDs. Do not infer patient identity solely from the filenames.

## Explanations and robustness

For a newly trained checkpoint:

```powershell
python -m medical_gradcam explain --image data/test/PNEUMONIA/person158_bacteria_744.jpeg
python -m medical_gradcam explain --image data/test/PNEUMONIA/person158_bacteria_744.jpeg --method gradcam++
```

`--target NORMAL` or `--target PNEUMONIA` attributes a specific class; the default
attributes the predicted class. Outputs include `comparison.png`, `overlay.png`,
raw `heatmap.npy` and prediction metadata. Grad-CAM hooks are released after use.
Heatmaps are class attributions, not validated lesion boundaries.

Robustness uses a reproducible random sample and compares the original with:

- A border mask filled with the image's median color; width defaults to 8%.
- A central mask of equal pixel area as an occlusion control.
- Brightness multiplied by 0.9 and 1.1.
- A 5% horizontal translation.

It saves example perturbation images, per-image scores, prediction flip rates,
accuracy and changes in the probability of the original predicted class.
These interventions may remove meaningful features or create distribution shift.
Sensitivity alone does not establish shortcut learning. This version does not
perform lung segmentation, clinical report generation or clinical diagnosis.

## Evaluation in Jupyter

Open `Evaluation.ipynb` with the `.venv` Python kernel and run the cells. It uses
the Python modules to display training curves (when available), test metrics,
the confusion matrix, Grad-CAM comparisons and robustness results inline.
It never starts training. It loads `run_dir/best.pt`; train in the terminal
before running the notebook. Check the printed checkpoint before evaluating.

## Where model weights are stored

With `pretrained: true`, torchvision loads ImageNet weights from its shared
cache (usually `~/.cache/torch/hub/checkpoints`), downloading them if needed.
The classifier is replaced with a two-class layer, then all model parameters
are fine-tuned on your training images.

Old experiment outputs and trained weights were cleared for a fresh start.
Module-based training saves its best checkpoint to `runs/baseline/best.pt` (or your configured `run_dir`). That
file is updated whenever validation accuracy improves, or validation loss improves
at equal accuracy. It stores weights and metadata, not an extracted model folder.

## Modules and checks

| Module | Purpose |
|---|---|
| `config.py` | Settings, seeds and device selection |
| `data.py` | Transforms, image datasets, persistent splits and patient groups |
| `model.py` | ResNet creation and checkpoint loading |
| `train.py` | Training and validation |
| `evaluate.py` | Classification and calibration metrics |
| `explain.py` | Grad-CAM and Grad-CAM++ artifacts |
| `robustness.py` | Perturbation experiments |

```powershell
python -m unittest discover -s tests -v
```

Tests use small synthetic images and random model weights; they make no downloads
or API calls. They check split isolation, verified patient overlap rejection,
checkpoint provenance and a one-epoch end-to-end pipeline. The baseline follows
the existing notebook and standard torchvision ResNet / pytorch-grad-cam usage.
