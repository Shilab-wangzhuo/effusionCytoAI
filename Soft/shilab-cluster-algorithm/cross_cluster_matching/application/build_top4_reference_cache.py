#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a reference cache using the Top4 reference-domain workflow only.

This is intentionally independent of ``build_reference_cache.py`` and of the
production ``pipeline.py``.  It contains the same reference-domain sequence as
the Top4 notebook:

    collect_reference_data -> cell-size feature -> feature extraction
    -> PCA.fit_transform -> KMeans.fit -> Top4 UMAP visualisations

The GUI JSON configuration is used only as the single source of truth for the
model, reference paths and preprocessing values.  Candidate folders in that
JSON are deliberately ignored.
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
SOURCE_NAMES = {1: "LUAD", 2: "LUSC", 3: "SCLC", 4: "Benign", 5: "Malignant"}


def set_top4_seed() -> None:
    """Use the same seed settings as the Top4 notebook."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float_list(value: Any, field: str) -> list[float]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} 必须是逗号分隔字符串或数值列表。")
    return [float(item) for item in value]


def parse_int_list(value: Any, field: str) -> list[int]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} 必须是逗号分隔字符串或整数列表。")
    return [int(item) for item in value]


def load_top4_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = (
        "malignant_cells_dir", "benign_cells_dir", "model_path", "model_type",
        "feature_layer", "reference_k", "pca_dim", "batch_size", "num_workers",
        "cell_size_weight", "mean", "std", "malignant_ref_clusters",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"配置缺少字段：{', '.join(missing)}")

    result = dict(config)
    for name in ("reference_k", "pca_dim", "batch_size", "num_workers"):
        result[name] = int(result[name])
    result["cell_size_weight"] = float(result["cell_size_weight"])
    result["mean"] = parse_float_list(result["mean"], "mean")
    result["std"] = parse_float_list(result["std"], "std")
    result["malignant_ref_clusters"] = parse_int_list(
        result["malignant_ref_clusters"], "malignant_ref_clusters"
    )
    return result


def ensure_new_cache_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"输出目录已存在且非空：{output_dir}\n"
            "Top4 cache 不允许覆盖。请使用新的、带版本号的目录。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def get_root_and_relative(path: str, malignant_dir: Path, benign_dir: Path) -> tuple[str, str]:
    absolute_path = Path(path).resolve()
    for root_name, root in (("malignant", malignant_dir), ("benign", benign_dir)):
        try:
            return root_name, absolute_path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    raise ValueError(f"参考图不在配置的 malignant/benign 路径内：{path}")


def save_top4_cache(
    output_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    *,
    ref_paths: list[str],
    ref_sources: np.ndarray,
    ref_sizes: np.ndarray,
    ref_features: np.ndarray,
    ref_normalized_features: np.ndarray,
    pca: PCA,
    kmeans: KMeans,
    umap_by_cluster: Any | None,
    umap_by_category: Any | None,
    hash_images: bool,
) -> None:
    """Persist exactly the objects made by this Top4 run; never re-fit here."""
    malignant_dir = Path(config["malignant_cells_dir"])
    benign_dir = Path(config["benign_cells_dir"])
    model_path = Path(config["model_path"])
    labels = np.asarray(kmeans.labels_, dtype=np.int64)
    ref_sources = np.asarray(ref_sources, dtype=np.int64)
    ref_sizes = np.asarray(ref_sizes, dtype=np.float64)
    ref_features = np.asarray(ref_features)
    ref_normalized_features = np.asarray(ref_normalized_features)
    if not (len(ref_paths) == len(ref_sources) == len(ref_sizes) == len(labels)):
        raise ValueError("参考图、来源标签、尺寸或聚类标签数量不一致。")
    if ref_features.shape[0] != len(ref_paths) or ref_normalized_features.shape[0] != len(ref_paths):
        raise ValueError("参考特征矩阵行数与参考图数量不一致。")

    manifest_rows: list[dict[str, Any]] = []
    for index, (image_path, source_label, cluster_label) in enumerate(
        tqdm(zip(ref_paths, ref_sources, labels), total=len(ref_paths), desc="保存 reference manifest")
    ):
        path = Path(image_path)
        root_name, relative_path = get_root_and_relative(image_path, malignant_dir, benign_dir)
        manifest_rows.append({
            "reference_index": index,
            "source_root": root_name,
            "relative_path": relative_path,
            "source_label": int(source_label),
            "source_name": SOURCE_NAMES.get(int(source_label), f"Unknown_{source_label}"),
            "cluster_label": int(cluster_label),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path) if hash_images else "",
        })
    manifest = pd.DataFrame(manifest_rows)

    np.savez_compressed(
        output_dir / "reference_features.npz",
        raw_features=ref_features,
        projected_features=pca.transform(ref_features),
        normalized_features=ref_normalized_features,
        source_labels=ref_sources,
        cluster_labels=labels,
        reference_sizes=ref_sizes,
        cluster_centers=kmeans.cluster_centers_,
    )
    np.save(output_dir / "cluster_centers.npy", kmeans.cluster_centers_)
    joblib.dump(pca, output_dir / "pca.joblib")
    joblib.dump(kmeans, output_dir / "kmeans.joblib")
    manifest.to_csv(output_dir / "reference_manifest.csv", index=False, encoding="utf-8-sig")

    composition = (
        manifest.groupby(["cluster_label", "source_name"], sort=True)
        .size().unstack(fill_value=0).sort_index()
    )
    composition.to_csv(output_dir / "Reference_cluster_composition.csv", encoding="utf-8-sig")

    saved_umap: dict[str, str] = {}
    for name, reducer in (("by_cluster", umap_by_cluster), ("by_category", umap_by_category)):
        if reducer is None:
            continue
        joblib_path = output_dir / f"reference_umap_{name}.joblib"
        joblib.dump(reducer, joblib_path)
        embedding = getattr(reducer, "embedding_", None)
        if embedding is not None:
            np.save(output_dir / f"reference_umap_{name}_embedding.npy", embedding)
        saved_umap[name] = joblib_path.name

    metadata = {
        "schema_version": 1,
        "cache_kind": "top4_reference_domain",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(config_path.resolve()),
        "model": {
            "model_type": config["model_type"],
            "model_path_at_build": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "feature_layer": config["feature_layer"],
        },
        "preprocessing": {"mean": config["mean"], "std": config["std"]},
        "reference_parameters": {
            "reference_k": int(kmeans.n_clusters),
            "requested_pca_dim": int(config["pca_dim"]),
            "effective_pca_dim": int(pca.n_components_),
            "batch_size": int(config["batch_size"]),
            "num_workers": int(config["num_workers"]),
            "cell_size_weight": float(config["cell_size_weight"]),
            "reference_path_order": "Top4 collect_reference_data runtime order",
            "image_sha256_written": hash_images,
        },
        "kmeans": {
            "random_state": getattr(kmeans, "random_state", None),
            "requested_n_init": getattr(kmeans, "n_init", None),
            "effective_n_init": getattr(kmeans, "_n_init", None),
            "algorithm": getattr(kmeans, "algorithm", None),
        },
        "malignant_ref_clusters": config["malignant_ref_clusters"],
        "reference_counts": {
            "total": int(len(manifest)),
            "malignant": int((manifest["source_root"] == "malignant").sum()),
            "benign": int((manifest["source_root"] == "benign").sum()),
        },
        "saved_umap": saved_umap,
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


def build_top4_reference_cache(
    config_path: Path,
    output_dir: Path,
    *,
    batch_size_override: int | None,
    write_visualizations: bool,
    hash_images: bool,
) -> None:
    config = load_top4_config(config_path)
    if batch_size_override is not None:
        config["batch_size"] = batch_size_override
    ensure_new_cache_dir(output_dir)
    set_top4_seed()

    for field in ("malignant_cells_dir", "benign_cells_dir", "model_path"):
        if not Path(config[field]).exists():
            raise FileNotFoundError(f"{field} 不存在：{config[field]}")

    print("=" * 72)
    print("Top4 参考域聚类 + reference_cache 导出")
    print(f"配置文件       : {config_path}")
    print(f"输出 cache     : {output_dir}")
    print(f"模型           : {config['model_type']}")
    print(f"模型权重       : {config['model_path']}")
    print(f"参考图路径     : malignant={config['malignant_cells_dir']}")
    print(f"                 benign={config['benign_cells_dir']}")
    print(f"mean / std     : {config['mean']} / {config['std']}")
    print(f"batch / PCA / K: {config['batch_size']} / {config['pca_dim']} / {config['reference_k']}")
    print(f"恶性参考簇     : {config['malignant_ref_clusters']}")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, _ = get_model_and_transform(
        config["model_type"], device, config["model_path"], mean=config["mean"], std=config["std"]
    )

    # This block deliberately mirrors the reference-domain block in the Top4
    # notebook, including its nested log1p cell-size behaviour.
    ref_paths, ref_sources = collect_reference_data(
        config["malignant_cells_dir"], config["benign_cells_dir"]
    )
    if not ref_paths:
        raise RuntimeError("没有找到参考图。")
    ref_sizes = np.log1p([
        count_non_background_pixels(path) for path in tqdm(ref_paths, desc="Ref Sizes")
    ])
    ref_loader = DataLoader(
        CellImageDataset(ref_paths, transform=transform, source_labels=ref_sources),
        batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
    )
    ref_feats, ref_sources, ref_paths = extract_features_with_cell_size(
        model, ref_loader, device, ref_sizes, config["feature_layer"], config["cell_size_weight"]
    )
    pca = PCA(n_components=min(config["pca_dim"], ref_feats.shape[0], ref_feats.shape[1]))
    ref_norm = normalize(pca.fit_transform(ref_feats))
    # Keep precisely the Top4 constructor; do not add a separate pipeline
    # reproducibility policy in this Top4-only script.
    kmeans = KMeans(n_clusters=config["reference_k"], random_state=42).fit(ref_norm)

    umap_by_cluster = None
    umap_by_category = None
    if write_visualizations:
        visualization_dir = output_dir / "reference_visualization"
        visualization_dir.mkdir(exist_ok=True)
        umap_by_cluster = visualize_all_clusters_without_images(
            ref_norm, kmeans.labels_, ref_sources, "Reference", str(visualization_dir)
        )
        umap_by_category = visualize_all_clusters_without_images_1(
            ref_norm, kmeans.labels_, ref_sources, "Reference", str(visualization_dir)
        )
        improved_visualize_clusters_with_images(
            ref_norm, kmeans.labels_, ref_sources, ref_paths, "Reference",
            str(visualization_dir), ref_sizes,
        )
        analyze_cluster_composition(
            kmeans.labels_, ref_sources, ref_paths, "Reference", str(visualization_dir)
        )

    save_top4_cache(
        output_dir, config, config_path,
        ref_paths=ref_paths,
        ref_sources=ref_sources,
        ref_sizes=ref_sizes,
        ref_features=ref_feats,
        ref_normalized_features=ref_norm,
        pca=pca,
        kmeans=kmeans,
        umap_by_cluster=umap_by_cluster,
        umap_by_category=umap_by_category,
        hash_images=hash_images,
    )
    print(f"\n完成：Top4 reference_cache 已保存到 {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="独立运行 Top4 参考域聚类并保存 reference_cache。"
    )
    parser.add_argument("--config", required=True, type=Path, help="Top4 对应的 pipeline_gui JSON 配置")
    parser.add_argument("--output_dir", required=True, type=Path, help="新的、空的 Top4 cache 输出目录")
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="覆盖 JSON batch_size；如文章 Top4 原运行使用 8，则传入 --batch_size 8。",
    )
    parser.add_argument("--write_visualizations", action="store_true", help="生成 Top4 风格参考域图并保存 UMAP")
    parser.add_argument("--no_hash_images", action="store_true", help="不计算参考图 SHA256（不推荐）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_top4_reference_cache(
        args.config, args.output_dir,
        batch_size_override=args.batch_size,
        write_visualizations=args.write_visualizations,
        hash_images=not args.no_hash_images,
    )


if __name__ == "__main__":
    main()
