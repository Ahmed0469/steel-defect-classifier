# Steel Surface Defect Classifier

A CNN built from scratch (Keras/TensorFlow) that classifies steel surface
defects into 6 categories, with a documented investigation into
preprocessing choices and their effect on class-level confusion.

## Problem

Classify steel surface images into one of 6 defect types:
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.

## Dataset

NEU-style steel surface defect dataset — 6 classes, balanced (60 images per
class in the test set). Images are stored as 3-channel RGB but are true
grayscale underneath (confirmed by checking that all 3 channels are
pixel-identical) — this informed the grayscale-loading decision below.
Dataset link :https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

## Pipeline

1. **Data quality checks** — verified class balance, checked for corrupted
   files, verified images were effectively grayscale despite being stored
   as RGB.
2. **CLAHE preprocessing** (`src/apply_clahe.py`) — Contrast Limited
   Adaptive Histogram Equalization applied to enhance local texture
   contrast, which matters for defects distinguished mainly by fine surface
   texture rather than color.
3. **Grayscale loading** — since color channels carried no extra
   information, images are loaded as single-channel to remove redundant
   input and slightly reduce model size.
4. **CNN training** (`src/train.py`) — 3 Conv blocks with Dropout,
   EarlyStopping, and ReduceLROnPlateau.
5. **Evaluation** (`src/evaluate.py`) — classification report + confusion
   matrix on a held-out test set.

## Key findings

- **Suspiciously unstable training was traced to the learning rate, not the
  architecture.** Increasing the learning rate 10x during an iteration
  (0.001 → 0.01) caused loss to get stuck at `ln(num_classes)` — the
  signature of a model whose weights aren't meaningfully updating because
  the optimizer keeps overshooting. Reverting fixed it immediately.
- **A generic data-augmentation attempt (random rotation, contrast,
  brightness, flip) significantly hurt performance** (accuracy dropped from
  82% to 48%). Root cause: `scratches` and `crazing`/`inclusion` are
  partly distinguished by directional texture — random rotation destroys
  exactly the feature that separates a linear scratch from a random-texture
  crazing pattern. Lesson: augmentation should be chosen based on what
  actually differentiates the classes, not applied by default.
- **CLAHE's clip limit mattered more than expected.** `clipLimit=2.0`
  improved crazing/inclusion/scratches discrimination but introduced a new
  confusion — `rolled-in_scale` (already a directional/textured defect)
  started being misclassified as `scratches`, because the stronger contrast
  enhancement made its texture look more line-like. Lowering to
  `clipLimit=1.5` kept the gains without this regression.
- **`inclusion` vs `pitted_surface` confusion persisted across every
  architecture and preprocessing variant tried** (grayscale conversion,
  deeper Conv stack, CLAHE at two different clip limits). This consistency
  suggests either a genuinely hard visual distinction between these two
  defect types, or some inherent ambiguity in how the dataset labels them
  — a limitation worth stating plainly rather than tuning around
  indefinitely.

## Results

Final model (grayscale + 3 Conv blocks + CLAHE clipLimit=1.5):

```
                 precision    recall  f1-score   support
        crazing       0.94      1.00      0.97        60
      inclusion       0.87      0.68      0.77        60
        patches       1.00      0.98      0.99        60
 pitted_surface       0.77      0.95      0.85        60
rolled-in_scale       1.00      1.00      1.00        60
      scratches       0.95      0.88      0.91        60
       accuracy                           0.92       360
```

Progression across iterations:

| Version | Accuracy | Notes |
|---|---|---|
| Baseline RGB CNN | 82% | crazing over-predicted (recall 1.00, precision 0.66) |
| + grayscale, 3rd Conv block | 86% | fixed crazing over-prediction |
| + CLAHE (clip=2.0) | 87% | introduced rolled-in_scale/scratches confusion |
| + CLAHE (clip=1.5) | **92%** | resolved the clip=2.0 regression |

## Next steps

- Manually inspect `inclusion`/`pitted_surface` misclassified examples to
  determine if the confusion reflects genuine visual ambiguity or
  borderline labeling.
- Compare against a fine-tuned pretrained backbone (MobileNetV3) to see
  whether richer pretrained features resolve the persistent
  inclusion/pitted_surface confusion.
- Try augmentation types that respect the directional nature of some
  defects (e.g., horizontal flip only, no rotation).

## How to run

```bash
pip install -r requirements.txt

# 1. Preprocess data with CLAHE
python src/apply_clahe.py --input Data/train --output Data/train_clahe
python src/apply_clahe.py --input Data/test --output Data/test_clahe

# 2. Train
python src/train.py

# 3. Evaluate
python src/evaluate.py
```

## Data
Currently sample test images are placed inside data/test_clahe you can evaluate on this
test images but if you want to use your own data place the raw dataset under `Data/train/<class_name>/*.jpg` and
`Data/test/<class_name>/*.jpg` before running the CLAHE preprocessing step.
Raw and processed data are excluded from this repo (see `.gitignore`) —
only a small sample is included under `data/sample/` for a quick pipeline
check.
Dataset link :https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database