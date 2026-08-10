#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an immutable reference-domain cache for cross-cluster matching.

This command runs *only* the reference-domain part of ``pipeline.py`` once,
then serialises every object needed by later candidate-domain runs.  It is
intended for a reviewed, versioned reference domain: never overwrite a cache
that has already been used for analysis/publication.

Example (run in the local environment that produced the reviewed reference
result)::

    python -m cross_cluster_matching.application.build_reference_cache \
        --config J:/limr/effusion/code/clustering_config/final/\
pipeline_gui_config_20260714_0.9.json \
        --output_dir J:/limr/effusion/result/reference_cache/\
single_cell_densenet161_pca32_v1 \
        --write_visualizations

The cache contains the fitted PCA and KMeans objects, fixed reference cluster
centres, the fixed image-to-cluster mapping, a manifest and an audit trail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from cross_cluster_matching.data.data_utils import CellImageDataset, collect_reference_data
from cross_cluster_matching.models.feature_extractor import extract_features_with_cell_size
from cross_cluster_matching.models.models import get_model_and_transform
from cross_cluster_matching.visualization.visualization import (
    analyze_cluster_composition,
    count_non_background_pixels,
    improved_visualize_clusters_with_images,
    visualize_all_clusters_without_images,
    visualize_all_clusters_without_images_1,
)


SEED = 42
SOURCE_LABEL_NAMES = {1: "LUAD", 2: "LUSC", 3: "SCLC", 4: "benign", 5: "malignant"}


def _set_seed() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv_floats(value: Any, name: str) -> list[float]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"配置项 {name} 必须是逗号分隔字符串或数值列表。")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置项 {name} 含有无法转换为 float 的值：{value!r}") from exc


def _parse_clusters(value: Any) -> list[int]:
    if isinstance(value, str):
        value = value.split(",")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malignant_ref_clusters 配置错误：{value!r}") from exc


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = [
        "malignant_cells_dir", "benign_cells_dir", "model_path", "model_type",
        "feature_layer", "reference_k", "pca_dim", "batch_size", "num_workers",
        "cell_size_weight", "mean", "std", "malignant_ref_clusters",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"配置缺少必要字段：{', '.join(missing)}")

    parsed = dict(config)
    for key in ("reference_k", "pca_dim", "batch_size", "num_workers"):
        parsed[key] = int(parsed[key])
    parsed["cell_size_weight"] = float(parsed["cell_size_weight"])
    parsed["mean"] = _parse_csv_floats(parsed["mean"], "mean")
    parsed["std"] = _parse_csv_floats(parsed["std"], "std")
    parsed["malignant_ref_clusters"] = _parse_clusters(parsed["malignant_ref_clusters"])
    return parsed


def _validate_reference_images(image_paths: list[str]) -> None:
    """Fail rather than silently replacing a broken reference image with black."""
    failures: list[str] = []
    for image_path in tqdm(image_paths, desc="验证参考图", unit="image"):
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:  # PIL exposes several format-specific exceptions.
            failures.append(f"{image_path}: {exc}")
            if len(failures) >= 10:
                break
    if failures:
        detail = "\n  - ".join(failures)
        raise RuntimeError(
            "参考域中存在无法读取的图像；为避免生成污染的 cache，已停止。\n"
            f"  - {detail}"
        )


def _canonical_path_info(image_path: str, malignant_dir: Path, benign_dir: Path) -> tuple[str, str]:
    path = Path(image_path).resolve()
    for source, root in (("malignant", malignant_dir), ("benign", benign_dir)):
        try:
            return source, path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    raise ValueError(f"参考图不在 malignant 或 benign 目录内：{image_path}")


def _build_manifest(
    image_paths: list[str], source_labels: np.ndarray, labels: np.ndarray,
    malignant_dir: Path, benign_dir: Path, hash_images: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, (image_path, source_label, cluster_label) in enumerate(
        tqdm(zip(image_paths, source_labels, labels), total=len(image_paths), desc="写入参考清单", unit="image")
    ):
        path = Path(image_path)
        source_root, relative_path = _canonical_path_info(image_path, malignant_dir, benign_dir)
        records.append({
            "reference_index": index,
            "source_root": source_root,
            "relative_path": relative_path,
            "source_label": int(source_label),
            "source_name": SOURCE_LABEL_NAMES.get(int(source_label), f"unknown_{source_label}"),
            "cluster_label": int(cluster_label),
            "file_size_bytes": path.stat().st_size,
            "sha256": _sha256(path) if hash_images else "",
        })
    return pd.DataFrame.from_records(records)


def _ensure_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"输出 cache 目录已存在且非空：{output_dir}\n"
            "reference_cache 是版本化证据，脚本拒绝覆盖。请改用新的 v2 目录。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def build_reference_cache(
    config: dict[str, Any], config_path: Path, output_dir: Path,
    *, sort_reference_paths: bool, hash_images: bool, write_visualizations: bool,
    pca_svd_solver: str, kmeans_n_init: int, kmeans_algorithm: str,
) -> None:
    _ensure_empty_output_dir(output_dir)
    _set_seed()

    malignant_dir = Path(config["malignant_cells_dir"])
    benign_dir = Path(config["benign_cells_dir"])
    model_path = Path(config["model_path"])
    for path, description in ((malignant_dir, "malignant_cells_dir"), (benign_dir, "benign_cells_dir"), (model_path, "model_path")):
        if not path.exists():
            raise FileNotFoundError(f"{description} 不存在：{path}")

    print("=" * 72)
    print("构建 reference_cache（只运行参考域；不会处理任何候选样本）")
    print(f"配置文件 : {config_path}")
    print(f"输出目录 : {output_dir}")
    print(f"模型     : {config['model_type']} / {model_path}")
    print(f"PCA / K  : {config['pca_dim']} / {config['reference_k']}")
    print(f"恶性簇   : {config['malignant_ref_clusters']}")
    print("=" * 72)

    ref_paths, ref_sources = collect_reference_data(str(malignant_dir), str(benign_dir))
    if sort_reference_paths:
        pairs = sorted(zip(ref_paths, ref_sources), key=lambda pair: Path(pair[0]).as_posix().casefold())
        ref_paths, ref_sources = map(list, zip(*pairs)) if pairs else ([], [])
    if not ref_paths:
        raise RuntimeError("未收集到任何参考图，请检查 malignant_cells_dir 和 benign_cells_dir。")
    if len(ref_paths) < config["reference_k"]:
        raise ValueError(f"参考图数量 {len(ref_paths)} 小于 reference_k={config['reference_k']}。")

    _validate_reference_images(ref_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, _ = get_model_and_transform(
        config["model_type"], device, str(model_path), mean=config["mean"], std=config["std"]
    )

    # Keep the existing pipeline's size-feature calculation exactly: it computes
    # log1p here, and extract_features_with_cell_size applies log1p once more.
    ref_sizes = np.log1p([
        count_non_background_pixels(path) for path in tqdm(ref_paths, desc="Ref Sizes")
    ])
    ref_loader = DataLoader(
        CellImageDataset(ref_paths, transform=transform, source_labels=ref_sources),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
    )
    ref_feats, ref_sources_array, ordered_paths = extract_features_with_cell_size(
        model, ref_loader, device, ref_sizes, config["feature_layer"], config["cell_size_weight"]
    )

    effective_pca_dim = min(config["pca_dim"], ref_feats.shape[0], ref_feats.shape[1])
    pca = PCA(n_components=effective_pca_dim, svd_solver=pca_svd_solver)
    projected_features = pca.fit_transform(ref_feats)
    ref_norm = normalize(projected_features)
    kmeans = KMeans(
        n_clusters=config["reference_k"], random_state=SEED,
        n_init=kmeans_n_init, algorithm=kmeans_algorithm,
    ).fit(ref_norm)

    ref_sources_array = np.asarray(ref_sources_array, dtype=np.int64)
    labels = np.asarray(kmeans.labels_, dtype=np.int64)
    ref_sizes = np.asarray(ref_sizes, dtype=np.float64)
    manifest = _build_manifest(
        ordered_paths, ref_sources_array, labels, malignant_dir, benign_dir, hash_images
    )

    # Store arrays separately from sklearn objects so a future loader can verify
    # shapes without deserialising pickle/joblib files.
    np.savez_compressed(
        output_dir / "reference_features.npz",
        raw_features=ref_feats,
        projected_features=projected_features,
        normalized_features=ref_norm,
        source_labels=ref_sources_array,
        cluster_labels=labels,
        reference_sizes=ref_sizes,
        cluster_centers=kmeans.cluster_centers_,
    )
    np.save(output_dir / "cluster_centers.npy", kmeans.cluster_centers_)
    joblib.dump(pca, output_dir / "pca.joblib")
    joblib.dump(kmeans, output_dir / "kmeans.joblib")
    manifest.to_csv(output_dir / "reference_manifest.csv", index=False, encoding="utf-8-sig")

    cluster_summary = (
        manifest.groupby(["cluster_label", "source_name"], sort=True)
        .size().unstack(fill_value=0).sort_index()
    )
    cluster_summary.to_csv(output_dir / "Reference_cluster_composition.csv", encoding="utf-8-sig")

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed reference domain for cross-cluster matching.",
        "source_config": str(config_path.resolve()),
        "model": {
            "model_type": config["model_type"],
            "model_path_at_build": str(model_path.resolve()),
            "sha256": _sha256(model_path),
            "feature_layer": config["feature_layer"],
        },
        "preprocessing": {"mean": config["mean"], "std": config["std"]},
        "reference_parameters": {
            "reference_k": config["reference_k"],
            "requested_pca_dim": config["pca_dim"],
            "effective_pca_dim": int(effective_pca_dim),
            "cell_size_weight": config["cell_size_weight"],
            "path_order": "sorted" if sort_reference_paths else "legacy_collect_reference_data_order",
            "image_sha256_written": hash_images,
        },
        "pca": {"svd_solver": pca_svd_solver},
        "kmeans": {"random_state": SEED, "n_init": kmeans_n_init, "algorithm": kmeans_algorithm},
        "malignant_ref_clusters": config["malignant_ref_clusters"],
        "reference_counts": {
            "total": int(len(manifest)),
            "malignant": int((manifest["source_root"] == "malignant").sum()),
            "benign": int((manifest["source_root"] == "benign").sum()),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with (output_dir / "source_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    if write_visualizations:
        visualization_dir = output_dir / "validation_visualizations"
        visualization_dir.mkdir(exist_ok=True)
        visualize_all_clusters_without_images(
            ref_norm, labels, ref_sources_array, "Reference", str(visualization_dir)
        )
        visualize_all_clusters_without_images_1(
            ref_norm, labels, ref_sources_array, "Reference", str(visualization_dir)
        )
        improved_visualize_clusters_with_images(
            ref_norm, labels, ref_sources_array, ordered_paths, "Reference",
            str(visualization_dir), ref_sizes,
        )
        analyze_cluster_composition(
            labels, ref_sources_array, ordered_paths, "Reference", str(visualization_dir)
        )

    print("\nreference_cache 构建完成。请先人工核对：")
    print(f"  - {output_dir / 'Reference_cluster_composition.csv'}")
    print("  - 若使用 --write_visualizations，再核对 validation_visualizations 中的图。")
    print("确认与文章参考域一致后，将整个 cache 目录只读备份并上传至服务器。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 pipeline_gui JSON 构建固定的参考域 reference_cache。"
    )
    parser.add_argument("--config", required=True, type=Path, help="pipeline_gui 导出的 JSON 配置文件")
    parser.add_argument("--output_dir", required=True, type=Path, help="新的、空的 cache 输出目录")
    parser.add_argument(
        "--sort_reference_paths", action="store_true",
        help=("按路径排序后再提取参考特征。首次复现历史文章结果时不要添加；"
              "新建、重新定义的参考域版本建议添加。"),
    )
    parser.add_argument("--no_hash_images", action="store_true", help="不计算每张参考图 SHA256（不推荐）")
    parser.add_argument("--write_visualizations", action="store_true", help="额外生成用于人工核对的参考域图")
    parser.add_argument("--pca_svd_solver", default="auto", choices=["auto", "full", "arpack", "randomized"])
    parser.add_argument("--kmeans_n_init", type=int, default=10)
    parser.add_argument("--kmeans_algorithm", default="lloyd", choices=["lloyd", "elkan"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    build_reference_cache(
        config, args.config, args.output_dir,
        sort_reference_paths=args.sort_reference_paths,
        hash_images=not args.no_hash_images,
        write_visualizations=args.write_visualizations,
        pca_svd_solver=args.pca_svd_solver,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_algorithm=args.kmeans_algorithm,
    )


if __name__ == "__main__":
    main()
