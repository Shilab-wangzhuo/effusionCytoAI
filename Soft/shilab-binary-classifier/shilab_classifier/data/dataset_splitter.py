"""Dataset preparation utilities."""

import os
import shutil
import random
from pathlib import Path
import pandas as pd
from datetime import datetime

VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def clear_directory(directory: str):
    """Clear and recreate a directory."""
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


def split_dataset(benign_path, malignant_path, output_dir,
                  split_ratio=0.9, random_seed=42):
    """Split image data into the configured subsets."""
    rng = random.Random(random_seed)

    if not os.path.exists(benign_path):
        raise ValueError(f"Benign image path does not exist: {benign_path}")
    if not os.path.exists(malignant_path):
        raise ValueError(f"Malignant image path does not exist: {malignant_path}")

    split_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    split_logs = []

    train_val_benign_dir    = os.path.join(output_dir, "train_val", "benign")
    train_val_malignant_dir = os.path.join(output_dir, "train_val", "malignant")
    test_benign_dir         = os.path.join(output_dir, "test",      "benign")
    test_malignant_dir      = os.path.join(output_dir, "test",      "malignant")

    print("=" * 50)
    print("Step 1: Clearing target directories...")
    for directory in [train_val_benign_dir, train_val_malignant_dir,
                      test_benign_dir, test_malignant_dir]:
        clear_directory(directory)

    print("\n" + "=" * 50)
    print("Step 2: Processing benign images (all assigned to train/val set)...")
    benign_log = _process_images(
        benign_path, train_val_benign_dir, test_benign_dir,
        split_ratio=1.0, rng=rng
    )
    split_logs.append({
        'Split Time':      split_time,
        'Class':           'Benign',
        'Input Path':      benign_path,
        'Output Path':     output_dir,
        'Split Ratio':     '100% -> Train/Val',
        'Train/Val Count': benign_log['train_val'],
        'Test Count':      benign_log['test'],
        'Total':           benign_log['total']
    })

    print("\n" + "=" * 50)
    print(f"Step 3: Processing malignant images (train/val ratio={split_ratio*100:.1f}%)...")
    malignant_log = _process_images(
        malignant_path, train_val_malignant_dir, test_malignant_dir,
        split_ratio=split_ratio, rng=rng
    )
    split_logs.append({
        'Split Time':      split_time,
        'Class':           'Malignant',
        'Input Path':      malignant_path,
        'Output Path':     output_dir,
        'Split Ratio':     f'{split_ratio*100:.1f}% -> Train/Val',
        'Train/Val Count': malignant_log['train_val'],
        'Test Count':      malignant_log['test'],
        'Total':           malignant_log['total']
    })

    print("\n" + "=" * 50)
    train_val_dir = os.path.join(output_dir, "train_val")
    test_dir      = os.path.join(output_dir, "test")
    print(f"Dataset split complete.")
    print(f"  Train/val set saved to: {train_val_dir}")
    print(f"  Test set saved to:      {test_dir}")

    df = pd.DataFrame(split_logs)
    excel_path = os.path.join(output_dir, "dataset_split_log.xlsx")
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"  Log saved to:           {excel_path}")

    return {
        'benign':    benign_log,
        'malignant': malignant_log,
        'log_path':  excel_path
    }


def _process_images(source_dir, train_val_dir, test_dir, split_ratio, rng):
    """Process images."""
    image_files = sorted([
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"  ⚠️  Warning: no image files found in {source_dir}")
        return {'train_val': 0, 'test': 0, 'total': 0}

    file_groups: dict[str, list[str]] = {}
    for file in image_files:
        ext = os.path.splitext(file)[1].lower()
        file_groups.setdefault(ext, []).append(file)

    train_val_files = []
    test_files      = []
    detail_logs     = []

    for ext, files in sorted(file_groups.items()):
        rng.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        tv = files[:split_idx]
        te = files[split_idx:]
        train_val_files.extend(tv)
        test_files.extend(te)
        detail_logs.append(
            f"  ext {ext:6s}: total={len(files):4d}, "
            f"train/val={len(tv):4d}, test={len(te):4d}"
        )

    print(f"  Stratified sampling: train/val={len(train_val_files)}, "
          f"test={len(test_files)}, total={len(image_files)}")
    print("  Breakdown by extension:")
    for log in detail_logs:
        print(log)

    def safe_copy(file: str, dst_dir: str):
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(dst_dir, file)
        if os.path.exists(dst_path):
            raise FileExistsError(
                f"\n❌ Filename conflict!\n"
                f"   Destination: {dst_path}\n"
                f"   Source:      {src_path}\n"
            )
        shutil.copy2(src_path, dst_path)

    for file in train_val_files:
        safe_copy(file, train_val_dir)
    for file in test_files:
        safe_copy(file, test_dir)

    print(f"  ✅ File copy complete.")

    return {
        'train_val': len(train_val_files),
        'test':      len(test_files),
        'total':     len(image_files)
    }