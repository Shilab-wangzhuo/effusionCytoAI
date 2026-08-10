"""Example: complete 5-class training and evaluation pipeline."""

import os
from datetime import datetime

import torch

from shilab_cancer_classifier import (
    CANCER_CLASSES,
    NUM_CLASSES,
    SUPPORTED_MODELS,
    evaluate_all_models,
    prepare_cross_validation,
    split_dataset,
    train_all_models,
)


if __name__ == "__main__":
    raw_data_dir = r"H:\0UR_classification_model\raw_data"
    base_output_dir = r"H:\0UR_classification_model"
    experiment_dir = os.path.join(
        base_output_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    split_data_dir = os.path.join(experiment_dir, "split_data")
    cv_data_dir = os.path.join(experiment_dir, "cross_validation_data")
    models_dir = os.path.join(experiment_dir, "train_val_models")
    results_dir = os.path.join(experiment_dir, "evaluation_results")
    train_val_dir = os.path.join(split_data_dir, "train_val")
    test_dir = os.path.join(split_data_dir, "test")

    missing_classes = [
        class_name for class_name in CANCER_CLASSES
        if not os.path.exists(os.path.join(raw_data_dir, class_name))
    ]
    if missing_classes:
        raise FileNotFoundError(f"Missing class folders: {missing_classes}")

    models_to_train = ['ResNet50']
    # models_to_train = SUPPORTED_MODELS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    split_dataset(
        raw_data_dir=raw_data_dir,
        class_names=CANCER_CLASSES,
        output_dir=split_data_dir,
        split_ratio=0.9,
        random_seed=42,
    )

    prepare_cross_validation(
        data_path=train_val_dir,
        output_base=cv_data_dir,
        n_splits=5,
        random_state=42,
    )

    train_all_models(
        model_list=models_to_train,
        cv_data_dir=cv_data_dir,
        output_dir=models_dir,
        num_folds=5,
        num_epochs=50,
        batch_size=16,
        initial_lr=0.001,
        device=device,
        num_classes=NUM_CLASSES,
        num_workers=0,
        patience=10,
    )

    evaluate_all_models(
        model_list=models_to_train,
        cv_data_dir=cv_data_dir,
        test_dir=test_dir,
        models_dir=models_dir,
        output_dir=results_dir,
        num_folds=5,
        batch_size=16,
        num_workers=0,
        num_classes=NUM_CLASSES,
        device=device,
        save_summary=True,
    )

    print(f"Pipeline complete: {experiment_dir}")
