"""Example: split the 5-class cancer image dataset."""

from shilab_cancer_classifier import CANCER_CLASSES, split_dataset


if __name__ == "__main__":
    raw_data_dir = r"path/to/raw_data"
    output_dir = r"path/to/output"

    result = split_dataset(
        raw_data_dir=raw_data_dir,
        class_names=CANCER_CLASSES,
        output_dir=output_dir,
        split_ratio=0.9,
        random_seed=42,
    )

    print("\n=== Split result ===")
    for class_name in CANCER_CLASSES:
        print(f"{class_name}: {result[class_name]}")
    print(f"Log file: {result['log_path']}")
