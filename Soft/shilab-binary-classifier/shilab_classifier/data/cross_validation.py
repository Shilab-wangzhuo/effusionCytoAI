"""Dataset preparation utilities."""

import os
import pandas as pd
from sklearn.model_selection import KFold
from glob import glob
import shutil
from datetime import datetime

from .dataset_splitter import clear_directory


VALID_IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']


def prepare_cross_validation(data_path: str, output_base: str, n_splits: int = 5, random_state: int = 42):
    """Prepare cross-validation folds."""
    print(f"Preparing {n_splits}-fold cross-validation data...")
    print(f"Data path:   {data_path}")
    print(f"Output path: {output_base}")

    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")

    image_files = []
    for ext in VALID_IMAGE_EXTENSIONS:
        image_files.extend(glob(os.path.join(data_path, '**', ext), recursive=True))
    image_files = sorted(image_files)

    if not image_files:
        raise ValueError(f"No image files found in {data_path}")
    print(f"Found {len(image_files)} images")

    file_paths = []
    classes    = []
    for img_path in image_files:
        class_name = os.path.basename(os.path.dirname(img_path))
        file_paths.append(img_path)
        classes.append(class_name)

    df = pd.DataFrame({'file_path': file_paths, 'class': classes})
    print(f"\nClass distribution:")
    print(df['class'].value_counts())

    print(f"\nClearing output directory...")
    clear_directory(output_base)

    kf             = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    unique_classes = sorted(df['class'].unique())
    fold_logs      = []
    split_time     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\nSplitting into {n_splits} folds...")
    print("=" * 50)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        fold_dir  = os.path.join(output_base, f'fold_{fold + 1}')
        train_dir = os.path.join(fold_dir, 'train')
        val_dir   = os.path.join(fold_dir, 'val')

        for cls in unique_classes:
            clear_directory(os.path.join(train_dir, cls))
            clear_directory(os.path.join(val_dir,   cls))

        train_data = df.iloc[train_idx]
        val_data   = df.iloc[val_idx]

        def safe_copy(src: str, dst: str):
            if os.path.exists(dst):
                raise FileExistsError(
                    f"\n❌ Filename conflict!\n"
                    f"   Destination: {dst}\n"
                    f"   Source:      {src}"
                )
            shutil.copy2(src, dst)

        for _, row in train_data.iterrows():
            src = row['file_path']
            dst = os.path.join(train_dir, row['class'], os.path.basename(src))
            safe_copy(src, dst)

        for _, row in val_data.iterrows():
            src = row['file_path']
            dst = os.path.join(val_dir, row['class'], os.path.basename(src))
            safe_copy(src, dst)

        train_dist = train_data['class'].value_counts().to_dict()
        val_dist   = val_data['class'].value_counts().to_dict()

        row_data = {'Fold': f'fold_{fold + 1}'}
        for cls in unique_classes:
            row_data[f'Training_{cls}']   = train_dist.get(cls, 0)
            row_data[f'Validation_{cls}'] = val_dist.get(cls, 0)
        row_data['Training_Total']   = len(train_data)
        row_data['Validation_Total'] = len(val_data)
        fold_logs.append(row_data)

        print(f"Fold {fold + 1}: train={len(train_data)}, val={len(val_data)}")

    df_summary = pd.DataFrame(fold_logs).set_index('Fold')

    col_map = {}
    for cls in unique_classes:
        col_map[f'Training_{cls}']   = ('Training Set',   cls.capitalize())
        col_map[f'Validation_{cls}'] = ('Validation Set', cls.capitalize())
    col_map['Training_Total']   = ('Training Set',   'Total')
    col_map['Validation_Total'] = ('Validation Set', 'Total')

    df_display = df_summary.rename(columns=col_map)
    df_display.columns = pd.MultiIndex.from_tuples(df_display.columns)

    print("\n" + "=" * 50)
    print("Fold split summary:")
    print("=" * 50)
    print(df_display.to_string())
    print("=" * 50)

    log_path = os.path.join(output_base, "cross_validation_log.xlsx")
    with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
        df_display.to_excel(writer, sheet_name='Fold Summary')

        df_detail = pd.DataFrame(fold_logs)
        df_detail.insert(1, 'Split Time', split_time)
        df_detail.to_excel(writer, sheet_name='Detail', index=False)

    print(f"\n{n_splits}-fold cross-validation dataset ready.")
    print(f"Output directory: {output_base}")
    print(f"Log saved to:     {log_path}")

    return {
        'n_splits':   n_splits,
        'total':      len(df),
        'log_path':   log_path,
        'df_summary': df_display,
    }