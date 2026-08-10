"""Example: train selected models for the 5-class cancer classifier."""

import torch

from shilab_cancer_classifier import NUM_CLASSES, SUPPORTED_MODELS, train_all_models


if __name__ == "__main__":
    cv_data_dir = r"path/to/cross_validation_data"
    output_dir = r"path/to/trained_models"

    model_list = [
        'ResNet50',
        'VGG16',
        'DenseNet121',
        'EfficientNetB0',
    ]
    # model_list = SUPPORTED_MODELS

    result = train_all_models(
        model_list=model_list,
        cv_data_dir=cv_data_dir,
        output_dir=output_dir,
        num_folds=5,
        num_epochs=50,
        batch_size=32,
        initial_lr=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_classes=NUM_CLASSES,
        num_workers=4,
        patience=10,
        save_log=True,
    )

    print("\nTraining complete.")
    print(f"Total time: {result['total_time']:.2f} minutes")
    print(f"Output dir: {result['output_dir']}")
