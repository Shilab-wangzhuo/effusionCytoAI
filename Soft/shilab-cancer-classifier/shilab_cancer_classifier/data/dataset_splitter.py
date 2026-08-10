"""Dataset splitting utilities for multi-class image classification."""

import os
import random
import shutil
from datetime import datetime

import pandas as pd

from shilab_cancer_classifier.config import CANCER_CLASSES


VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def clear_directory(directory: str):
    """Empty a directory while keeping the directory itself."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"  Cleared directory: {directory}")
    else:
        os.makedirs(directory)
        print(f"  Created directory: {directory}")


def split_dataset(raw_data_dir=None, output_dir=None, class_names=None,
                  split_ratio=0.9, random_seed=42, class_paths=None,
                  benign_path=None, malignant_path=None):
    """
    Split a class-folder image dataset into train_val and test folders.

    Expected multi-class input:
        raw_data_dir/
            Breast/
            GI_Tract/
            GYN/
            Lung_cancer/
            Mesothelioma/

    Backward-compatible binary input is still accepted through benign_path and
    malignant_path, but new code should use raw_data_dir/class_names.
    """
    if output_dir is None:
        raise ValueError("output_dir must be provided")

    rng = random.Random(random_seed)
    split_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if class_paths is None:
        if raw_data_dir is not None:
            class_names = class_names or CANCER_CLASSES
            class_paths = {class_name: os.path.join(raw_data_dir, class_name)
                           for class_name in class_names}
        elif benign_path is not None and malignant_path is not None:
            class_paths = {'benign': benign_path, 'malignant': malignant_path}
        else:
            raise ValueError("Provide raw_data_dir or class_paths")

    for class_name, source_dir in class_paths.items():
        if not os.path.exists(source_dir):
            raise ValueError(f"Class directory does not exist for {class_name}: {source_dir}")

    print("=" * 50)
    print("Step 1: Clearing target directories...")
    for class_name in class_paths:
        clear_directory(os.path.join(output_dir, "train_val", class_name))
        clear_directory(os.path.join(output_dir, "test", class_name))

    split_logs = []
    results = {}

    print("\n" + "=" * 50)
    print(f"Step 2: Splitting {len(class_paths)} classes with train_val ratio {split_ratio:.3f}...")
    for class_name, source_dir in class_paths.items():
        train_val_dir = os.path.join(output_dir, "train_val", class_name)
        test_dir = os.path.join(output_dir, "test", class_name)
        print(f"\nClass: {class_name}")
        class_log = _process_images(source_dir, train_val_dir, test_dir, split_ratio, rng)
        results[class_name] = class_log
        split_logs.append({
            'split_time': split_time,
            'class': class_name,
            'input_path': source_dir,
            'output_path': output_dir,
            'train_val_ratio': split_ratio,
            'train_val_count': class_log['train_val'],
            'test_count': class_log['test'],
            'total_count': class_log['total'],
        })

    train_val_dir = os.path.join(output_dir, "train_val")
    test_dir = os.path.join(output_dir, "test")
    print("\n" + "=" * 50)
    print("Dataset split complete.")
    print(f"  Train/val data: {train_val_dir}")
    print(f"  Test data:      {test_dir}")

    df = pd.DataFrame(split_logs)
    excel_path = os.path.join(output_dir, "dataset_split_log.xlsx")
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"  Log saved to:   {excel_path}")

    results['log_path'] = excel_path
    return results


def _process_images(source_dir, train_val_dir, test_dir, split_ratio, rng):
    image_files = sorted([
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"  Warning: no image files found in {source_dir}")
        return {'train_val': 0, 'test': 0, 'total': 0}

    file_groups = {}
    for file in image_files:
        ext = os.path.splitext(file)[1].lower()
        file_groups.setdefault(ext, []).append(file)

    train_val_files = []
    test_files = []

    for ext, files in sorted(file_groups.items()):
        rng.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        tv = files[:split_idx]
        te = files[split_idx:]
        train_val_files.extend(tv)
        test_files.extend(te)
        print(f"  {ext:6s}: total={len(files):4d}, train_val={len(tv):4d}, test={len(te):4d}")

    def safe_copy(file: str, dst_dir: str):
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(dst_dir, file)
        if os.path.exists(dst_path):
            raise FileExistsError(f"File name collision: {dst_path}")
        shutil.copy2(src_path, dst_path)

    for file in train_val_files:
        safe_copy(file, train_val_dir)
    for file in test_files:
        safe_copy(file, test_dir)

    print(f"  Copied {len(train_val_files)} train_val and {len(test_files)} test images.")
    return {
        'train_val': len(train_val_files),
        'test': len(test_files),
        'total': len(image_files),
    }
