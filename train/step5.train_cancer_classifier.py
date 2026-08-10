"""
Multi-class cancer classification training pipeline.
Includes dataset splitting, cross-validation, training, and evaluation.

"""

import os
from pathlib import Path
from datetime import datetime

import torch
from shilab_cancer_classifier import (
    SUPPORTED_MODELS,
    evaluate_all_models,
    prepare_cross_validation,
    split_dataset,
    train_all_models,
    set_seed,
)

# ── Class config ──────────────────────────────
CANCER_CLASSES = [
    "Bile_duct", "Breast", "Cervix", "Colorectum", "Endometrium",
    "Esophagus", "Gastric", "Mesothelioma", "NSCLC", "Ovary",
    "Pancreas", "SCLC",
]
NUM_CLASSES = len(CANCER_CLASSES)

# ── Path config ───────────────────────────────
RAW_DATA_DIR   = Path(r"/path/to/your/raw_data")
BASE_OUTPUT_DIR = Path(r"/path/to/your/output")

# ── Experiment directories ────────────────────
EXPERIMENT_DIR = BASE_OUTPUT_DIR
SPLIT_DATA_DIR = EXPERIMENT_DIR / "split_data"
CV_DATA_DIR    = EXPERIMENT_DIR / "cross_validation_data"
MODELS_DIR     = EXPERIMENT_DIR / "train_val_models"
RESULTS_DIR    = EXPERIMENT_DIR / "evaluation_results"
TRAIN_VAL_DIR  = SPLIT_DATA_DIR / "train_val"
TEST_DIR       = SPLIT_DATA_DIR / "test"

# ── Training hyperparameters ──────────────────
SEED        = 42
NUM_FOLDS   = 5
NUM_EPOCHS  = 50
BATCH_SIZE  = 16
INITIAL_LR  = 0.001
PATIENCE    = 10
NUM_WORKERS = 0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_TO_TRAIN = SUPPORTED_MODELS  # or specify a subset, e.g. ["ResNet50"]

# ── Pipeline switches ─────────────────────────
RUN_SPLIT = True
RUN_CV    = True
RUN_TRAIN = True
RUN_EVAL  = True


def validate_data(raw_data_dir: Path, cancer_classes: list, num_folds: int) -> None:
    """Check that all class folders exist and contain enough images."""
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    missing = [c for c in cancer_classes if not (raw_data_dir / c).exists()]
    if missing:
        raise FileNotFoundError(f"Missing class folders: {missing}")

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for class_name in cancer_classes:
        count = sum(
            1 for p in (raw_data_dir / class_name).iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        )
        if count < num_folds:
            raise ValueError(
                f"Class '{class_name}' has only {count} images, "
                f"which is less than NUM_FOLDS={num_folds}."
            )
    print("Data validation passed.")


def main():
    set_seed(SEED)

    # Create output directories
    for d in [EXPERIMENT_DIR, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Device       : {DEVICE}")
    print(f"Models       : {MODELS_TO_TRAIN}")
    print(f"Classes ({NUM_CLASSES}): {CANCER_CLASSES}")
    print(f"Experiment   : {EXPERIMENT_DIR}")

    # ── Step 0: Validate input data ───────────────
    validate_data(RAW_DATA_DIR, CANCER_CLASSES, NUM_FOLDS)

    # ── Step 1: Split dataset ─────────────────────
    if RUN_SPLIT:
        split_result = split_dataset(
            raw_data_dir=str(RAW_DATA_DIR),
            class_names=CANCER_CLASSES,
            output_dir=str(SPLIT_DATA_DIR),
            split_ratio=0.9,
            random_seed=SEED,
        )
        print(f"Dataset split complete. Log: {split_result['log_path']}")
    else:
        print("Skipping split step.")

    # ── Step 2: Prepare cross-validation folds ────
    if RUN_CV:
        cv_result = prepare_cross_validation(
            data_path=str(TRAIN_VAL_DIR),
            output_base=str(CV_DATA_DIR),
            n_splits=NUM_FOLDS,
            random_state=SEED,
        )
        print(f"Cross-validation ready. Log: {cv_result['log_path']}")
    else:
        print("Skipping cross-validation step.")

    # ── Step 3: Train models ──────────────────────
    if RUN_TRAIN:
        print(f"Training {len(MODELS_TO_TRAIN)} model(s) × {NUM_FOLDS} folds "
              f"| start: {datetime.now():%Y-%m-%d %H:%M:%S}")
        train_result = train_all_models(
            model_list=MODELS_TO_TRAIN,
            cv_data_dir=str(CV_DATA_DIR),
            output_dir=str(MODELS_DIR),
            num_folds=NUM_FOLDS,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            initial_lr=INITIAL_LR,
            device=DEVICE,
            num_classes=NUM_CLASSES,
            num_workers=NUM_WORKERS,
            patience=PATIENCE,
        )
        print(f"Training complete. Output: {train_result['output_dir']}")
    else:
        print("Skipping training step.")

    # ── Step 4: Evaluate models ───────────────────
    if RUN_EVAL:
        print(f"Evaluating models | start: {datetime.now():%Y-%m-%d %H:%M:%S}")
        eval_result = evaluate_all_models(
            model_list=MODELS_TO_TRAIN,
            cv_data_dir=str(CV_DATA_DIR),
            test_dir=str(TEST_DIR),
            models_dir=str(MODELS_DIR),
            output_dir=str(RESULTS_DIR),
            num_folds=NUM_FOLDS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            save_summary=True,
        )
        if eval_result:
            print(f"Evaluation complete. Total time: {eval_result['total_time']:.2f} min")
        else:
            print("Evaluation returned no results.")
    else:
        print("Skipping evaluation step.")


if __name__ == "__main__":
    main()