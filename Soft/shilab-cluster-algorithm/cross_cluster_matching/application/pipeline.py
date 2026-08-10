# pipeline.py

# 导入所有整理好的模块 - 使用当前的实际模块化结构
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import json
import hashlib
from pathlib import Path
import joblib
import umap
from tqdm import tqdm

from cross_cluster_matching.models.models import get_model_and_transform
from cross_cluster_matching.data.data_utils import collect_reference_data, collect_image_paths, CellImageDataset, extract_patient_id
from cross_cluster_matching.models.feature_extractor import extract_features_with_cell_size
from cross_cluster_matching.clustering.K_optimizer import optimize_k_for_candidate_clustering
from cross_cluster_matching.clustering.calculate_consensus_score import calculate_consensus_score
from cross_cluster_matching.clustering.matching_rule_application import identify_matching_clusters, save_matched_cells, check_matching_rules
from cross_cluster_matching.visualization.visualization import (
    count_non_background_pixels,
    visualize_all_clusters_without_images, 
    visualize_all_clusters_without_images_1, 
    improved_visualize_clusters_with_images, 
    visualize_matching_clusters,
    analyze_cluster_composition,
    plot_similarity_heatmap,
    visualize_filtered_clusters
)

# 设置随机种子，确保结果可复现
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _sha256_file(path):
    """Return the SHA256 recorded in reference_cache metadata."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_cache_value(name, actual, expected):
    if actual != expected:
        raise ValueError(
            f"reference_cache 参数不匹配：{name}，cache={actual!r}，当前配置={expected!r}"
        )


def _assert_cache_float_list(name, actual, expected):
    if expected is None:
        return
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array, expected_array, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"reference_cache 参数不匹配：{name}，cache={actual_array.tolist()}，"
            f"当前配置={expected_array.tolist()}"
        )


def load_reference_cache(
    reference_cache_dir, model_path, model_type, feature_layer,
    mean, std, reference_k, pca_dim, cell_size_weight, malignant_ref_clusters,
    reference_umap_mode="runtime_refit",
):
    """Load a fixed Top4 reference cache without touching reference images.

    ``runtime_refit`` keeps all inference-critical cache artefacts fixed, then
    fits the UMAP visualisation reducer in the current Python process.  A
    serialised UMAP reducer contains Numba internals and is not portable across
    otherwise similar local and server environments.
    """
    cache_dir = Path(reference_cache_dir)
    required_files = {
        "metadata": cache_dir / "metadata.json",
        "pca": cache_dir / "pca.joblib",
        "kmeans": cache_dir / "kmeans.joblib",
        "features": cache_dir / "reference_features.npz",
        "manifest": cache_dir / "reference_manifest.csv",
    }
    if reference_umap_mode == "cached":
        required_files["umap"] = cache_dir / "reference_umap_by_cluster.joblib"
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"reference_cache 不完整：{cache_dir}，缺少 {', '.join(missing)}。"
        )

    with required_files["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"不支持的 reference_cache schema_version：{metadata.get('schema_version')!r}")

    cache_model = metadata.get("model", {})
    cache_preprocess = metadata.get("preprocessing", {})
    cache_parameters = metadata.get("reference_parameters", {})
    _assert_cache_value("model_type", cache_model.get("model_type"), model_type)
    _assert_cache_value("feature_layer", cache_model.get("feature_layer"), feature_layer)
    _assert_cache_float_list("mean", cache_preprocess.get("mean"), mean)
    _assert_cache_float_list("std", cache_preprocess.get("std"), std)
    _assert_cache_value("reference_k", int(cache_parameters.get("reference_k")), int(reference_k))
    _assert_cache_value("effective_pca_dim", int(cache_parameters.get("effective_pca_dim")), int(pca_dim))
    _assert_cache_value(
        "cell_size_weight", float(cache_parameters.get("cell_size_weight")), float(cell_size_weight)
    )
    _assert_cache_value(
        "malignant_ref_clusters", list(metadata.get("malignant_ref_clusters", [])),
        [int(cluster) for cluster in malignant_ref_clusters],
    )

    current_model_hash = _sha256_file(model_path)
    _assert_cache_value("model_sha256", cache_model.get("sha256"), current_model_hash)

    pca = joblib.load(required_files["pca"])
    kmeans = joblib.load(required_files["kmeans"])
    arrays = np.load(required_files["features"], allow_pickle=False)
    manifest = pd.read_csv(required_files["manifest"])
    features = np.asarray(arrays["normalized_features"])
    labels = np.asarray(arrays["cluster_labels"], dtype=int)
    source_labels = np.asarray(arrays["source_labels"], dtype=int)
    centers = np.asarray(arrays["cluster_centers"])
    n_reference = len(features)
    if not (len(labels) == len(source_labels) == len(manifest) == n_reference):
        raise ValueError("reference_cache 内部数组长度不一致，拒绝继续运行。")
    if centers.shape[0] != int(reference_k) or kmeans.n_clusters != int(reference_k):
        raise ValueError("reference_cache 的簇中心数量与 reference_k 不一致。")
    if not np.allclose(centers, kmeans.cluster_centers_, rtol=0.0, atol=1e-12):
        raise ValueError("reference_cache 中 cluster_centers 与 kmeans.joblib 不一致。")

    if reference_umap_mode == "runtime_refit":
        # Only the two-dimensional visualisation is re-fitted. The input is
        # the cache's frozen article-version reference features, so this does
        # not alter PCA, KMeans, reference labels/centres or matching results.
        print("[INFO] Re-fitting UMAP in the current process for visualisation only.")
        reference_umap = umap.UMAP(random_state=42).fit(features)
    elif reference_umap_mode == "cached":
        # Legacy option. Cross-environment deserialisation can fail only when
        # transform() first invokes Numba, even if joblib.load() succeeds.
        print("[WARN] Using serialised UMAP from cache; this is not portable across environments.")
        reference_umap = joblib.load(required_files["umap"])
    else:
        raise ValueError(
            f"Unknown reference_umap_mode={reference_umap_mode!r}; "
            "expected 'runtime_refit' or 'cached'."
        )

    image_paths = manifest["relative_path"].astype(str).tolist()
    ref_results = {
        "features": features,
        "labels": labels,
        "source_labels": source_labels,
        "image_paths": image_paths,
        "cluster_centers": centers,
        "n_clusters": int(reference_k),
        "pca": pca,
        "kmeans": kmeans,
    }
    print(f"[INFO] 已加载并校验 reference_cache: {cache_dir}")
    print(f"[INFO] 参考图数量={n_reference}，恶性参考簇={metadata['malignant_ref_clusters']}")
    return ref_results, reference_umap


def run_pipeline(
    positive_folder_path, negative_folder_path, malignant_cells_dir=None, benign_cells_dir=None,
    model_path=None, base_save_dir=None,
    feature_layer='penultimate', reference_k=10, max_candidate_k=20, 
    batch_size=16, num_workers=0, cell_size_weight=1.0, model_type='ResNeXt', pca_dim=32 , mean = None , std = None, malignant_ref_clusters=None, matching_rules=None,
    use_cell_level_rule_filter=False, reference_cache_dir=None,
    patient_id=None, stats_path=None, reference_umap_mode="runtime_refit",
    save_rule_diagnostics=False,
):
    """
    运行跨域簇匹配分析流水线，用于分析细胞图像中的恶性细胞与良性细胞的匹配关系
    - 实现参考域（已知恶性/良性细胞）与候选域（待分析病理切片）之间的簇匹配
    - 应用两种匹配规则识别恶性细胞亚群
    - 计算匹配簇的共识得分并生成可视化结果
    
    参数:
    positive_folder_path: 包含阳性病例文件夹的路径
    negative_folder_path: 包含阴性病例文件夹的路径
    malignant_cells_dir: 参考域恶性细胞图像存储路径
    benign_cells_dir: 参考域良性细胞图像存储路径
    model_path: 预训练模型文件路径
    base_save_dir: 结果保存的基础目录
    feature_layer: 用于特征提取的模型层，默认为'penultimate'
    reference_k: 参考域聚类数量，默认为10
    max_candidate_k: 候选域最大聚类数量，默认为20
    batch_size: 数据加载批次大小，默认为16
    num_workers: 数据加载进程数，默认为0
    cell_size_weight: 细胞大小权重因子，默认为1.0
    model_type: 使用的模型类型，默认为'ResNeXt'
    pca_dim: PCA降维后的维度，默认为32
    mean: 图像归一化的均值，默认为None
    std: 图像归一化的标准差，默认为None
    malignant_ref_clusters -恶性簇索引
    matching_rules : list of dict | None
        规则配置列表，每个 dict 描述一条规则，字段如下：
            {
                "target"               : int,             # 目标参考簇索引
                "avoid"                : int | list[int], # 需要避开的参考簇索引
                "threshold_target"     : float,           # 目标相似度阈值
                "threshold_avoid_others": float | None,   # 其他簇绝对阈值（可选）
                "threshold_ratio"      : float | None,    # 其他簇相对阈值（可选）
            }
        若为 None，则使用默认规则（基于 malignant_ref_clusters 自动生成）。
    
    use_cell_level_rule_filter : bool, default=True
        是否在聚类模式下，对候选匹配簇内的细胞继续使用 matching_rules 做细胞级严格过滤。
        True:
            save_matched_cells(..., rule=matching_rules[i])
            细胞不仅要求最相似参考簇属于 malignant_ref_clusters，
            还必须满足对应规则的 threshold_target / threshold_ratio / threshold_avoid_others 等条件。

        False:
            save_matched_cells(..., rule=None)
            细胞只要求最相似参考簇属于 malignant_ref_clusters，
            不再额外检查 matching_rules。

    """

    # ── 参数校验与默认规则生成 ──────────────────────────────────────────────
    if malignant_ref_clusters is None:
        raise ValueError("malignant_ref_clusters 不能为 None")
    if model_path is None or base_save_dir is None:
        raise ValueError("model_path 和 base_save_dir 不能为空")
    if reference_cache_dir is None and (not malignant_cells_dir or not benign_cells_dir):
        raise ValueError("未提供 reference_cache_dir 时，malignant_cells_dir 与 benign_cells_dir 均为必填。")

    n_rules = len(malignant_ref_clusters)  # 规则数量 = 恶性簇数量

    if matching_rules is None:
        # 默认规则：每个恶性簇为 target，其余恶性簇为 avoid
        matching_rules = [
            {
                "target": malignant_ref_clusters[i],
                "avoid": [c for j, c in enumerate(malignant_ref_clusters) if j != i],
                "threshold_target": 0.5,
                "threshold_ratio": 3,
            }
            for i in range(n_rules)
        ]

    assert len(matching_rules) == n_rules, \
        f"matching_rules 长度({len(matching_rules)})必须与 malignant_ref_clusters 长度({n_rules})一致"

    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 1. 初始化环境与模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, _ = get_model_and_transform(model_type, device, model_path, mean= mean, std = std)
    
    # 2. Reference Domain: load reviewed Top4 cache when supplied; otherwise
    # keep the original re-clustering path for legacy configurations.
    if reference_cache_dir:
        ref_results, reference_umap = load_reference_cache(
            reference_cache_dir, model_path, model_type, feature_layer,
            mean, std, reference_k, pca_dim, cell_size_weight, malignant_ref_clusters,
            reference_umap_mode=reference_umap_mode,
        )
        pca = ref_results['pca']
    else:
        print("======= 处理参考细胞 =======")
        ref_paths, ref_sources = collect_reference_data(malignant_cells_dir, benign_cells_dir)
        ref_sizes = np.log1p([count_non_background_pixels(p) for p in tqdm(ref_paths, desc="Ref Sizes")])
        ref_loader = DataLoader(
            CellImageDataset(ref_paths, transform=transform, source_labels=ref_sources),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        ref_feats, ref_sources, ref_paths = extract_features_with_cell_size(
            model, ref_loader, device, ref_sizes, feature_layer, cell_size_weight
        )
        pca = PCA(n_components=min(pca_dim, ref_feats.shape[0], ref_feats.shape[1]))
        ref_norm = normalize(pca.fit_transform(ref_feats))
        kmeans = KMeans(n_clusters=reference_k, random_state=42).fit(ref_norm)
        ref_results = {
            'features': ref_norm, 'labels': kmeans.labels_, 'source_labels': ref_sources,
            'image_paths': ref_paths, 'cluster_centers': kmeans.cluster_centers_,
            'n_clusters': reference_k, 'pca': pca
        }
        ref_vis_dir = os.path.join(base_save_dir, f'reference_visualization_pca{pca_dim}')
        os.makedirs(ref_vis_dir, exist_ok=True)
        reference_umap = visualize_all_clusters_without_images(
            ref_results['features'], ref_results['labels'], ref_results['source_labels'], "Reference", ref_vis_dir
        )
        visualize_all_clusters_without_images_1(
            ref_results['features'], ref_results['labels'], ref_results['source_labels'], "Reference", ref_vis_dir
        )
        improved_visualize_clusters_with_images(
            ref_results['features'], ref_results['labels'], ref_results['source_labels'],
            ref_results['image_paths'], "Reference", ref_vis_dir, ref_sizes
        )
        analyze_cluster_composition(
            ref_results['labels'], ref_results['source_labels'], ref_results['image_paths'], "Reference", ref_vis_dir
        )

    # 3. 准备病理片子列表
    patient_folders = []
    patient_folders.extend([(entry.path, 'positive') for entry in os.scandir(positive_folder_path) if entry.is_dir()])
    patient_folders.extend([(entry.path, 'negative') for entry in os.scandir(negative_folder_path) if entry.is_dir()])
    
    # ── 按 patient_id 去重，保留首次出现的记录 ──────────────────────────
    # 场景1（推理模式）：positive 和 negative 传同一路径，去重后每个样本只保留 'positive' 那条
    # 场景2（训练模式）：positive 和 negative 是不同路径，patient_id 不重叠，去重不影响任何数据
    seen_pids = set()
    unique_patient_folders = []
    for path, status in patient_folders:
        pid = extract_patient_id(path)
        if pid not in seen_pids:
            seen_pids.add(pid)
            unique_patient_folders.append((path, status))
        else:
            print(f"[INFO] 跳过重复样本: {pid} (路径: {path}, 标签: {status})")
    patient_folders = unique_patient_folders

    # 单样本模式：Slurm array 的每个任务只能处理自己指定的样本，
    # 防止并发任务读取/写入其他样本的候选域结果。
    if patient_id:
        patient_folders = [
            (path, status) for path, status in patient_folders
            if extract_patient_id(path) == patient_id
        ]
        if not patient_folders:
            raise ValueError(
                f"未在候选域输入目录中找到指定样本 patient_id={patient_id!r}"
            )
    
    results_stats = []
    
    # 4. 循环处理病理片子
    for folder, status in patient_folders:
        pid = extract_patient_id(folder)
        save_dir = os.path.join(base_save_dir, pid)
        os.makedirs(save_dir, exist_ok=True)
        
        # 收集数据
        cand_paths = collect_image_paths(os.path.join(folder, "malignant_images"))
        if not cand_paths: continue

        cand_sources = [0]*len(cand_paths)
            
        cand_sizes = np.log1p([count_non_background_pixels(p) for p in cand_paths])
        
        # 提取特征
        cand_loader = DataLoader(
            CellImageDataset(cand_paths, transform=transform, source_labels=cand_sources),
            batch_size=batch_size, shuffle=False
        )
        cand_feats, _, cand_paths = extract_features_with_cell_size(
            model, cand_loader, device, cand_sizes, feature_layer, cell_size_weight
        )
        
        # PCA Transform
        cand_norm = normalize(pca.transform(cand_feats))
        
        # ── 动态初始化：matched_counts 和 dirs ──────────────────────────────
        # 有几条规则就生成几个计数器，全部初始化为 0
        matched_counts = [0] * n_rules   # [count_r1, count_r2, ...]

        # 动态生成每条规则的保存目录
        dirs = {f'rule{i+1}': os.path.join(save_dir, f'rule{i+1}_matched_cells') for i in range(n_rules)}
        dirs['total'] = os.path.join(save_dir, 'matched_malignant_cells')
        output_dirs = [dirs['total']]
        if save_rule_diagnostics:
            output_dirs.extend(dirs[f'rule{i+1}'] for i in range(n_rules))
        for d in output_dirs:
            os.makedirs(d, exist_ok=True)
        
        if len(cand_norm) >= pca_dim:
            # A. 聚类模式
            cand_results, best_k = optimize_k_for_candidate_clustering(
                cand_norm, cand_sources, cand_paths, ref_results, save_dir, pid, max_candidate_k, malignant_ref_clusters
            )
            if not cand_results: continue

            '''
            cand_results = {
                'features': patient_candidate_normalized,
                'labels': patient_candidate_labels,(每个元素对应一个细胞的聚类标签)
                'distances': patient_candidate_distances,
                'source_labels': patient_candidate_source_labels, (有LUAD,LUSC,SCLC和Candidate, 前面三个对应reference domain, 后面一个对应candidate domain)
                'image_paths': patient_candidate_image_paths,
                'kmeans': patient_candidate_kmeans,
                'cluster_centers': patient_candidate_kmeans.cluster_centers_,
                'n_clusters': k
            }
            '''
            # 存一下具体的分类结果
            csv_data = {
                'labels': cand_results['labels'],
                'image_paths': cand_results['image_paths']
            }

            df_cluster = pd.DataFrame(csv_data)
            csv_save_path = os.path.join(save_dir, 'cluster_detail.csv')

            # index=False 表示不保存行索引(0,1,2...)
            # encoding='utf-8-sig' 可以防止中文路径在 Excel 中打开乱码
            df_cluster.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

            # 识别匹配簇：返回 n_rules 个簇列表 + similarity_matrix
            *rule_clusters_list, similarity_matrix = identify_matching_clusters(
                cand_results['cluster_centers'], ref_results['cluster_centers'], matching_rules
            )
            # rule_clusters_list[i] 存放满足第 i 条规则的候选簇编号列表

            cell_sims = cosine_similarity(cand_results['features'], ref_results['cluster_centers'])

            # ── 动态保存每条规则匹配的细胞 ──────────────────────────────────
            passed_indices_dict = {}   # 按规则分组，直接传给 visualize_filtered_clusters

            for i, rule_clusters in enumerate(rule_clusters_list):
                rule_name = f'rule{i+1}'
                indices   = [j for j, lbl in enumerate(cand_results['labels']) if lbl in rule_clusters]

                rejected_dir = None
                if save_rule_diagnostics:
                    rejected_dir = os.path.join(save_dir, f'rule{i+1}_rejected_cells')
                    os.makedirs(rejected_dir, exist_ok=True)
                
                cell_filter_rule = matching_rules[i] if use_cell_level_rule_filter else None

                count, kept_indices = save_matched_cells(
                    indices, cand_paths, cand_results['labels'], cell_sims, rule_name,
                    {'rule_dir': dirs[rule_name], 'total_dir': dirs['total']},
                    malignant_ref_clusters,
                    rule          = cell_filter_rule,
                    save_rejected = save_rule_diagnostics,
                    rejected_dir  = rejected_dir,
                    save_rule_copies = save_rule_diagnostics,
                )
                matched_counts[i] = count
                passed_indices_dict[rule_name] = kept_indices   # 按规则分组

            # ── 动态生成 matching_pairs ──────────────────────────────────────
            matching_pairs = []
            for rule, rule_clusters in zip(matching_rules, rule_clusters_list):
                target_ref = rule["target"]
                for cand_idx in rule_clusters:
                    matching_pairs.append(
                        (target_ref, cand_idx, similarity_matrix[cand_idx, target_ref])
                    )

            # ── 动态生成 rule_matched_clusters ──────────────────────────────
            rule_matched_clusters = {
                f'rule{i+1}': clusters for i, clusters in enumerate(rule_clusters_list)
            }

            calculate_consensus_score(ref_results, cand_results, matching_pairs, save_dir)

            visualize_matching_clusters(
                ref_results, cand_results, matching_pairs,
                save_dir, reference_umap, rule_matched_clusters, output_format="png"
            )
            visualize_all_clusters_without_images(
                cand_results['features'], cand_results['labels'], cand_results['source_labels'],
                pid, save_dir, reference_umap, output_format="png"
            )
            improved_visualize_clusters_with_images(
                cand_results['features'], cand_results['labels'], cand_results['source_labels'],
                cand_paths, pid, save_dir, cand_sizes, output_format="png"
            )
            visualize_filtered_clusters(
                ref_results, cand_results, matching_pairs, save_dir,
                rule_matched_clusters = rule_matched_clusters,
                target_ref_indices    = malignant_ref_clusters,
                passed_indices_dict   = passed_indices_dict,
            )
            xlabel = [f"C{i}" for i in range(cand_results['n_clusters'])]
            ylabel = [f"R{i}" for i in range(ref_results['n_clusters'])]
            plot_similarity_heatmap(
                similarity_matrix.T, xlabel, ylabel,
                f"{pid} Similarity Matrix",
                os.path.join(save_dir, 'similarity_heatmap.png'), save_tiff=False
            )

        else:
            # B. 单细胞模式
            cell_sims = cosine_similarity(cand_norm, ref_results['cluster_centers'])

            # 动态逐条规则筛选，并排除已被前面规则处理过的细胞
            processed = set()   # 记录已处理的细胞索引，保证每个细胞只归属一条规则
            for i, rule in enumerate(matching_rules):
                rule_name = f'rule{i+1}'
                indices = [
                    j for j, s in enumerate(cell_sims)
                    if j not in processed and check_matching_rules(
                        s,
                        ref_idx_target         = rule["target"],
                        ref_idx_avoid          = rule["avoid"],
                        threshold_target       = rule["threshold_target"],
                        threshold_avoid_others = rule.get("threshold_avoid_others"),
                        threshold_ratio        = rule.get("threshold_ratio"),
                    )
                ]
                count, newly_processed = save_matched_cells(
                    indices, cand_paths, [0] * len(cand_paths), cell_sims, rule_name,
                    {'rule_dir': dirs[rule_name], 'total_dir': dirs['total']},
                    malignant_ref_clusters,
                    save_rule_copies=save_rule_diagnostics,
                )
                matched_counts[i] = count
                processed.update(newly_processed)  # 将本轮处理的索引加入已处理集合

            xlabel = [f"Cell{i}" for i in range(len(cell_sims))]
            ylabel = [f"R{i}" for i in range(ref_results['n_clusters'])]
            plot_similarity_heatmap(
                cell_sims.T, xlabel, ylabel,
                f"Patient {pid} Cell Similarity Matrix",
                os.path.join(save_dir, 'Cell_similarity_heatmap.png'), save_tiff=False
            )

        # ── 动态记录统计信息 ─────────────────────────────────────────────────
        stat = {
            'Patient_ID': pid,
            'Status': status,
            **{f'Rule{i+1}': matched_counts[i] for i in range(n_rules)},  # Rule1, Rule2, ...
            'Total': sum(matched_counts)
        }
        results_stats.append(stat)
        print(f"Patient {pid} Done. Matched: {sum(matched_counts)}")

    # 保存统计结果。单样本并发模式传入样本专属路径，避免多个 array task
    # 同时覆盖 base_save_dir/stats.xlsx；批处理模式仍保持原有默认位置。
    final_stats_path = stats_path or os.path.join(base_save_dir, 'stats.xlsx')
    stats_parent = os.path.dirname(os.path.abspath(final_stats_path))
    if stats_parent:
        os.makedirs(stats_parent, exist_ok=True)
    pd.DataFrame(results_stats).to_excel(final_stats_path)

# =============================================================
# 命令行入口（追加到 pipeline.py 文件末尾）
# =============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="跨域簇匹配假阳性去除")

    # 路径
    parser.add_argument("--positive_folder_path", required=True)
    parser.add_argument("--negative_folder_path", required=True)
    parser.add_argument("--malignant_cells_dir",  default=None)
    parser.add_argument("--benign_cells_dir",     default=None)
    parser.add_argument(
        "--reference_cache_dir", default=None,
        help="固定 Top4 reference_cache 目录；提供后不再读取或重聚类参考图。"
    )
    parser.add_argument(
        "--reference_umap_mode", choices=("runtime_refit", "cached"),
        default="runtime_refit",
        help=(
            "UMAP strategy for a reference cache. runtime_refit (default) fits "
            "UMAP from frozen cache features in this process and affects only "
            "visualisation; cached uses the legacy serialised UMAP reducer."
        ),
    )
    parser.add_argument("--model_path",           required=True)
    parser.add_argument("--base_save_dir",        required=True)
    parser.add_argument(
        "--patient_id", default=None,
        help="仅处理此样本 ID；供单 SVS / Slurm array 模式使用。"
    )
    parser.add_argument(
        "--stats_path", default=None,
        help="统计 Excel 的显式输出路径；未提供时仍保存为 base_save_dir/stats.xlsx。"
    )

    # 模型与聚类
    parser.add_argument("--model_type",       default="ResNeXt")
    parser.add_argument("--feature_layer",    default="penultimate")
    parser.add_argument("--reference_k",      type=int,   default=10)
    parser.add_argument("--max_candidate_k",  type=int,   default=20)
    parser.add_argument("--pca_dim",          type=int,   default=32)
    parser.add_argument("--cell_size_weight", type=float, default=1.0)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--workers",          type=int,   default=0)

    # 归一化
    parser.add_argument("--mean", nargs="+", type=float, default=None)
    parser.add_argument("--std",  nargs="+", type=float, default=None)

    # 恶性簇索引（逗号分隔，如 "8,0"）
    parser.add_argument("--malignant_ref_clusters", required=True)

    # 规则字符串（@分隔多条规则，每条: target:avoid:thr_target:thr_ratio:thr_avoid）
    # 示例: "8:0:0.61:2:none@0:8:0.93:none:0.3"
    parser.add_argument("--rules_str", default="")

    # 细胞级规则过滤开关
    parser.add_argument("--use_cell_level_rule_filter", action="store_true", default=False)
    parser.add_argument("--save_rule_diagnostics", action="store_true", default=False,
                        help="Save rule-level matched/rejected image copies for debugging.")

    # 日志标识
    parser.add_argument("--cell_type", default="unknown")

    args = parser.parse_args()

    if not args.reference_cache_dir and (not args.malignant_cells_dir or not args.benign_cells_dir):
        parser.error("未提供 --reference_cache_dir 时，必须同时提供 --malignant_cells_dir 和 --benign_cells_dir。")

    # ── 解析 malignant_ref_clusters ───────────────────────────
    mal_clusters = [int(x.strip()) for x in args.malignant_ref_clusters.split(",")]

    # ── 解析 rules_str → matching_rules ───────────────────────
    # 格式: "target:avoid:thr_target:thr_ratio:thr_avoid@..."
    matching_rules = None
    if args.rules_str:
        matching_rules = []
        for rule_str in args.rules_str.split("@"):
            parts = [p.strip() for p in rule_str.split(":")]
            if len(parts) != 5:
                raise ValueError(f"规则格式错误（应为5个字段）: {rule_str}")
            target, avoid, thr_target, thr_ratio, thr_avoid = parts

            # avoid 支持多个（逗号分隔），如 "0,2"
            avoid_parsed = [int(x) for x in avoid.split(",")] if "," in avoid else int(avoid)

            matching_rules.append({
                "target":                int(target),
                "avoid":                 avoid_parsed,
                "threshold_target":      float(thr_target),
                "threshold_ratio":       float(thr_ratio)  if thr_ratio != "none"  else None,
                "threshold_avoid_others":float(thr_avoid)  if thr_avoid != "none"  else None,
            })

    print(f"[INFO] cell_type              = {args.cell_type}")
    print(f"[INFO] model_type             = {args.model_type}")
    print(f"[INFO] malignant_ref_clusters = {mal_clusters}")
    print(f"[INFO] reference_umap_mode   = {args.reference_umap_mode}")
    print(f"[INFO] use_cell_level_filter  = {args.use_cell_level_rule_filter}")
    print(f"[INFO] save_rule_diagnostics  = {args.save_rule_diagnostics}")
    print(f"[INFO] matching_rules         = {matching_rules}")

    run_pipeline(
        positive_folder_path        = args.positive_folder_path,
        negative_folder_path        = args.negative_folder_path,
        malignant_cells_dir         = args.malignant_cells_dir,
        benign_cells_dir            = args.benign_cells_dir,
        reference_cache_dir         = args.reference_cache_dir,
        model_path                  = args.model_path,
        base_save_dir               = args.base_save_dir,
        model_type                  = args.model_type,
        feature_layer               = args.feature_layer,
        reference_k                 = args.reference_k,
        max_candidate_k             = args.max_candidate_k,
        pca_dim                     = args.pca_dim,
        cell_size_weight            = args.cell_size_weight,
        batch_size                  = args.batch_size,
        num_workers                 = args.workers,
        mean                        = args.mean,
        std                         = args.std,
        malignant_ref_clusters      = mal_clusters,
        matching_rules              = matching_rules,
        use_cell_level_rule_filter  = args.use_cell_level_rule_filter,
        patient_id                  = args.patient_id,
        stats_path                  = args.stats_path,
        reference_umap_mode         = args.reference_umap_mode,
        save_rule_diagnostics        = args.save_rule_diagnostics,
    )
