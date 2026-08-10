#!/usr/bin/env python
"""Infer cancer type from step-4 ``matched_malignant_cells`` images.

The input may be one ``matched_malignant_cells`` directory, or a step-4 output
root containing one such directory per patient.  Only images in directories
named exactly ``matched_malignant_cells`` are read; ``rule*_matched_cells``
and other diagnostic output are deliberately excluded.

The class order, normalization mean, and normalization standard deviation must
be supplied explicitly and must be identical to those used during training.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from shilab_cancer_classifier.models.model_definitions import create_model
from shilab_cancer_classifier.training.preprocessing import (
    pad_to_square_299_transform,
    pad_to_square_transform,
)

MATCHED_DIR_NAME = "matched_malignant_cells"
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MatchedCellImage:
    """One cell image retained by the step-4 false-positive filter."""

    image_path: Path
    patient_id: str
    matched_dir: Path


class MatchedCellDataset(Dataset):
    """Dataset that retains row-to-source-image alignment during inference."""

    def __init__(self, records: Sequence[MatchedCellImage], transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return value.strip("._") or "unknown"


def _parse_class_names(values: Iterable[str]) -> list[str]:
    class_names = []
    for value in values:
        class_names.extend(part.strip() for part in value.split(",") if part.strip())
    if not class_names:
        raise ValueError("--classes was supplied but contains no class names.")
    if len(class_names) != len(set(class_names)):
        raise ValueError(f"Class names must be unique, received: {class_names}")
    return class_names


def collect_matched_malignant_cells(input_dir: str | Path) -> list[MatchedCellImage]:
    """Collect images from step-4 outputs without mixing in rule-level copies."""

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    matched_dirs = [root] if root.name == MATCHED_DIR_NAME else sorted(
        path for path in root.rglob(MATCHED_DIR_NAME) if path.is_dir()
    )
    if not matched_dirs:
        raise FileNotFoundError(
            f"No '{MATCHED_DIR_NAME}' directory was found below {root}. "
            "Pass either a single patient matched directory or the step-4 output root."
        )

    records: list[MatchedCellImage] = []
    seen_paths: set[Path] = set()
    for matched_dir in matched_dirs:
        patient_id = matched_dir.parent.name
        for image_path in sorted(matched_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            resolved = image_path.resolve()
            if resolved not in seen_paths:
                records.append(MatchedCellImage(resolved, patient_id, matched_dir.resolve()))
                seen_paths.add(resolved)

    if not records:
        raise ValueError(f"Found {len(matched_dirs)} matched directories but no supported images below them.")
    print(f"Found {len(records)} cell images in {len(matched_dirs)} '{MATCHED_DIR_NAME}' directories.")
    return records


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint is not a state_dict or a supported checkpoint dictionary.")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def load_model_weights(model, model_path: str | Path, device: str):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
    return model


def _copy_confident_predictions(results_df: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    confident = results_df[results_df["predicted_probability"] >= threshold]
    copied_rows = []
    for _, row in confident.iterrows():
        destination_dir = output_dir / "confident_predictions" / _safe_name(row["predicted_class"]) / _safe_name(row["patient_id"])
        destination_dir.mkdir(parents=True, exist_ok=True)
        source = Path(row["image_path"])
        destination = destination_dir / f"{source.stem}_{row['predicted_probability']:.4f}{source.suffix}"
        duplicate_index = 1
        while destination.exists():
            destination = destination_dir / (
                f"{source.stem}_{row['predicted_probability']:.4f}_{duplicate_index}{source.suffix}"
            )
            duplicate_index += 1
        shutil.copy2(source, destination)
        copied_rows.append({
            "image_path": str(source),
            "copied_path": str(destination),
            "patient_id": row["patient_id"],
            "predicted_class": row["predicted_class"],
            "predicted_probability": row["predicted_probability"],
        })
    pd.DataFrame(copied_rows, columns=[
        "image_path", "copied_path", "patient_id", "predicted_class", "predicted_probability",
    ]).to_csv(output_dir / "confident_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"Copied {len(copied_rows)} predictions with confidence >= {threshold:.3f}.")


def _save_summaries(results_df: pd.DataFrame, class_names: Sequence[str], output_dir: Path) -> None:
    class_counts = (
        results_df.groupby(["patient_id", "predicted_class"]).size().unstack(fill_value=0)
        .reindex(columns=class_names, fill_value=0).reset_index()
    )
    mean_probabilities = results_df.groupby("patient_id")[[f"prob_{_safe_name(name)}" for name in class_names]].mean().reset_index()
    patient_summary = class_counts.merge(mean_probabilities, on="patient_id", how="outer")
    patient_summary.to_csv(output_dir / "patient_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(len(class_names), 1, figsize=(10, max(5, 2.1 * len(class_names))), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, class_name in zip(axes, class_names):
        column = f"prob_{_safe_name(class_name)}"
        axis.hist(results_df[column], bins=20, color="steelblue", edgecolor="white")
        axis.set_title(class_name)
        axis.set_ylabel("Cells")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Predicted probability")
    fig.tight_layout()
    fig.savefig(output_dir / "class_probability_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def predict_matched_malignant_cells(
    *,
    model_name: str,
    model_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int = 16,
    num_workers: int = 0,
    confidence_threshold: float = 0.5,
    device: str | None = None,
) -> pd.DataFrame:
    """Classify step-4 retained cells and write cell- and patient-level results.

    ``class_names``, ``mean``, and ``std`` must match the training configuration
    exactly.  No sidecar metadata file is required.
    """

    weights_path = Path(model_path).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    class_names = list(class_names)
    if not class_names:
        raise ValueError(
            "Class names are required. Pass them in the exact ImageFolder training order."
        )
    if len(class_names) != len(set(class_names)):
        raise ValueError(f"Class names must be unique, received: {class_names}")
    mean = list(mean)
    std = list(std)
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            "Training normalization mean/std are required. Pass --mean R G B and --std R G B."
        )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    records = collect_matched_malignant_cells(input_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pad_transform = pad_to_square_299_transform if "inception" in model_name.lower() else pad_to_square_transform
    dataset = MatchedCellDataset(records, transforms.Compose([
        transforms.Lambda(pad_transform), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
    ]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = create_model(model_name, num_classes=len(class_names))
    try:
        model = load_model_weights(model, weights_path, device).to(device).eval()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Unable to load {weights_path.name} as a {len(class_names)}-class {model_name} model. "
            "Check --model and the exact training class order supplied by --classes."
        ) from exc

    logits_batches, probability_batches, prediction_batches, indices = [], [], [], []
    with torch.no_grad():
        for inputs, batch_indices in loader:
            outputs = model(inputs.to(device))
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0] if isinstance(outputs, tuple) else outputs
            probabilities = torch.softmax(logits, dim=1)
            logits_batches.append(logits.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())
            prediction_batches.append(torch.argmax(logits, dim=1).cpu().numpy())
            indices.extend(batch_indices.tolist())

    logits = np.concatenate(logits_batches)
    probabilities = np.concatenate(probability_batches)
    predictions = np.concatenate(prediction_batches)
    ordered_records = [records[index] for index in indices]
    if logits.shape[1] != len(class_names):
        raise RuntimeError(f"Model emitted {logits.shape[1]} logits for {len(class_names)} class names.")

    results = pd.DataFrame({
        "patient_id": [record.patient_id for record in ordered_records],
        "matched_dir": [str(record.matched_dir) for record in ordered_records],
        "image_path": [str(record.image_path) for record in ordered_records],
        "filename": [record.image_path.name for record in ordered_records],
        "predicted_idx": predictions,
        "predicted_class": [class_names[index] for index in predictions],
        "predicted_probability": probabilities[np.arange(len(predictions)), predictions],
    })
    for index, class_name in enumerate(class_names):
        safe_name = _safe_name(class_name)
        results[f"logit_{safe_name}"] = logits[:, index]
        results[f"prob_{safe_name}"] = probabilities[:, index]

    results.to_csv(output_path / "prediction_results.csv", index=False, encoding="utf-8-sig")
    _copy_confident_predictions(results, output_path, confidence_threshold)
    _save_summaries(results, class_names, output_path)
    print(f"Inference complete: {len(results)} cells, {results['patient_id'].nunique()} patients.")
    return results


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Classify step-4 matched_malignant_cells images with a cancer classifier.")
    parser.add_argument("-m", "--model", required=True, help="Model architecture used for training, e.g. DenseNet161")
    parser.add_argument("-w", "--weights", required=True, help="Trained .pth weights")
    parser.add_argument("-i", "--input", required=True, help="A matched_malignant_cells directory or step-4 output root")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--classes", nargs="+", required=True, help="Exact ImageFolder training order; separate names by spaces or commas")
    parser.add_argument("--mean", type=float, nargs=3, required=True, metavar=("R", "G", "B"), help="Training normalization mean")
    parser.add_argument("--std", type=float, nargs=3, required=True, metavar=("R", "G", "B"), help="Training normalization standard deviation")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-j", "--workers", type=int, default=0)
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Copy predicted images with max probability >= threshold")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    predict_matched_malignant_cells(
        model_name=args.model,
        model_path=args.weights,
        input_dir=args.input,
        output_dir=args.output,
        class_names=_parse_class_names(args.classes),
        mean=args.mean,
        std=args.std,
        batch_size=args.batch_size,
        num_workers=args.workers,
        confidence_threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
