# ShiLab Cancer Classifier

`shilab-cancer-classifier` provides dataset preparation, cross-validation, training, evaluation, and inference utilities for four-class cancer classification from cytology cell images.

## Installation

```bash
cd path/to/shilab-cancer-classifier
pip install -e .
```

The package uses PyTorch, torchvision, scikit-learn, pandas, and the dependencies listed in `requirements.txt`. Select a PyTorch/CUDA installation appropriate for the execution host.

## Training data

Training images must be arranged in one folder per cancer class:

```text
raw_data/
├── Gastrointestinal_Breast/
├── Gynecologic/
├── Lung/
└── Mesothelioma/
```

The class-folder order is part of the trained model. `ImageFolder` sorts folder names alphabetically; retain the exact resulting order with each model weight.

## Inference after false-positive removal

The inference command accepts either a single `matched_malignant_cells/` directory or a Step-4 output root containing one such directory per patient. It ignores diagnostic directories such as `rule1_matched_cells/`.

```bash
shilab-cancer-infer \
  --model ViT_L16 \
  --weights /path/to/vit_l16.pth \
  --input /path/to/step4_output \
  --output /path/to/cancer_inference \
  --classes Gastrointestinal_Breast Gynecologic Lung Mesothelioma \
  --mean <training_mean_r> <training_mean_g> <training_mean_b> \
  --std <training_std_r> <training_std_g> <training_std_b> \
  --threshold 0.5
```

`--classes`, `--mean`, and `--std` are required because they are properties of the existing weight file, not inferred from the input data. Supply the exact ImageFolder training order and the normalization values used during training. Outputs include per-cell predictions, per-patient summaries, and thresholded copies organised by predicted class.

## Training workflow

This repository is a reusable library and does not include a project-specific
training entry script. Use its public dataset, cross-validation, training, and
evaluation functions from a project workflow. The reference implementation is
`effusionCytoAI/train/step5.train_cancer_classifier.py`.

The class configuration in `shilab_cancer_classifier/config.py` is the official
four-class configuration and must remain identical to the training folders and
the number of model outputs. Record the per-fold normalization values written
to the training log; they are required for inference.

## Scope

False-positive removal and cross-domain cluster matching belong to the separate `shilab-cluster-algorithm` package. Keeping that workflow separate prevents an unnecessary runtime dependency on clustering software during cancer classification.
