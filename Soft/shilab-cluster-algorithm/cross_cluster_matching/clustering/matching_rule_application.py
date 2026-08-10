# clustering/matching_rule_application.py
# 用来存放匹配规则设置及应用函数

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import shutil

def check_matching_rules(similarities, ref_idx_target, ref_idx_avoid, threshold_target, threshold_avoid_others=None, threshold_ratio=None):
    """
    通用规则检查函数
    similarities: 当前对象与所有参考簇的相似度
    ref_idx_target: 目标参考簇索引 (如 R5 或 R0)
    ref_idx_avoid: 需要避开的参考簇索引 (如 R0 或 R5)，可以是单个索引或索引列表
    threshold_target: 目标相似度阈值
    threshold_avoid_others: 其他簇相似度阈值 (绝对值)
    threshold_ratio: 其他簇相似度阈值 (相对于目标的比例)
    """
    similarities = [round(float(s), 2) for s in similarities]
    if similarities[ref_idx_target] < threshold_target:
        return False
        
    # 将ref_idx_avoid转换为列表
    avoid_indices = ref_idx_avoid if isinstance(ref_idx_avoid, list) else [ref_idx_avoid]
    
    # 计算其他索引（排除目标索引和需要避开的索引）
    other_indices = [i for i in range(len(similarities)) if i != ref_idx_target and i not in avoid_indices]
    
    if threshold_ratio:
        limit = similarities[ref_idx_target] / threshold_ratio
        if any(similarities[i] >= limit for i in other_indices):
            return False
            
    if threshold_avoid_others:
        if any(similarities[i] >= threshold_avoid_others for i in other_indices):
            return False
            
    return True

# # 匹配规则应用
# def identify_matching_clusters(candidate_centers, reference_centers):
#     """识别符合规则1和规则2的候选簇"""
#     similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
#     rule1_clusters = []
#     rule2_clusters = []
    
#     for i, sims in enumerate(similarity_matrix):
#         # 规则1: R5 >= 0.5, 其他 < R5/3 (忽略R0)
#         if check_matching_rules(sims, 5, 0, 0.5, threshold_ratio=3):
#             rule1_clusters.append(i)
#         # 规则2: R0 >= 0.4, 其他 < 0.2 (忽略R5) - 仅当不满足规则1时
#         elif check_matching_rules(sims, 0, 5, 0.4, threshold_avoid_others=0.2):
#             rule2_clusters.append(i)
        
            
#     return rule1_clusters, rule2_clusters, similarity_matrix

def identify_matching_clusters(candidate_centers, reference_centers, rules):
    """
    通用：识别符合各规则的候选簇。

    Parameters
    ----------
    candidate_centers : array-like, shape (n_candidates, n_features)
    reference_centers : array-like, shape (n_references, n_features)
    rules : list of dict，每个 dict 描述一条规则，字段如下：
        {
            "target"               : int,               # 目标参考簇索引
            "avoid"                : int | list[int],   # 需要避开的参考簇索引
            "threshold_target"     : float,             # 目标相似度阈值
            "threshold_avoid_others": float | None,     # 其他簇绝对阈值
            "threshold_ratio"      : float | None,      # 其他簇相对阈值（目标/ratio）
        }

    Returns
    -------
    rule_clusters : list of list[int]
        长度与 rules 相同，每个子列表包含满足对应规则的候选簇索引。
        规则按优先级依次判断（满足前面规则的簇不再判断后续规则）。
    similarity_matrix : ndarray, shape (n_candidates, n_references)

    Examples
    --------
    # 等价于原来的 identify_matching_clusters（血液版）
    rules = [
        {"target": 5, "avoid": 0, "threshold_target": 0.5, "threshold_ratio": 3},
        {"target": 0, "avoid": 5, "threshold_target": 0.4, "threshold_avoid_others": 0.2},
    ]
    rule1_clusters, rule2_clusters, sim = identify_matching_clusters(cand, ref, rules)

    # 等价于原来的 identify_matching_clusters_urine
    rules = [
        {"target": 0, "avoid": [4, 9], "threshold_target": 0.5, "threshold_ratio": 3},
        {"target": 4, "avoid": [0, 9], "threshold_target": 0.5, "threshold_ratio": 3},
        {"target": 9, "avoid": [0, 4], "threshold_target": 0.5, "threshold_ratio": 3},
    ]
    rule1, rule2, rule3, sim = identify_matching_clusters(cand, ref, rules)
    """
    similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
    rule_clusters = [[] for _ in rules]

    for i, sims in enumerate(similarity_matrix):
        for rule_idx, rule in enumerate(rules):
            matched = check_matching_rules(
                sims,
                ref_idx_target=rule["target"],
                ref_idx_avoid=rule["avoid"],
                threshold_target=rule["threshold_target"],
                threshold_avoid_others=rule.get("threshold_avoid_others"),
                threshold_ratio=rule.get("threshold_ratio"),
            )
            if matched:
                rule_clusters[rule_idx].append(i)
                break  # 满足前面规则后不再判断后续规则

    return (*rule_clusters, similarity_matrix)

def identify_matching_clusters_urine(candidate_centers, reference_centers):
    """识别符合规则1和规则2的候选簇"""
    similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
    rule1_clusters = []
    rule2_clusters = []
    rule3_clusters = []
    
    for i, sims in enumerate(similarity_matrix):
        # 规则1: R5 >= 0.5, 其他 < R5/3 (忽略R0)
        if check_matching_rules(sims, 0, [4, 9], 0.5, threshold_ratio=3):
            rule1_clusters.append(i)
        # 规则2: R0 >= 0.4, 其他 < 0.2 (忽略R5) - 仅当不满足规则1时
        elif check_matching_rules(sims, 4, [0, 9], 0.5, threshold_ratio=3):
            rule2_clusters.append(i)
        elif check_matching_rules(sims, 9, [0, 4], 0.5, threshold_ratio=3):
            rule3_clusters.append(i)
            
    return rule1_clusters, rule2_clusters, rule3_clusters, similarity_matrix

# 匹配规则应用
def identify_matching_clusters_urine_v2(candidate_centers, reference_centers):
    """识别符合规则1和规则2的候选簇"""
    similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
    rule1_clusters = []
    rule2_clusters = []
    
    for i, sims in enumerate(similarity_matrix):
        # 规则1: R7 >= 0.5, 其他 < R7/3 (忽略R0)
        if check_matching_rules(sims, 7, 0, 0.5, threshold_ratio=3):
            rule1_clusters.append(i)
        # 规则2: R0 >= 0.4, 其他 < 0.2 (忽略R7) - 仅当不满足规则1时
        elif check_matching_rules(sims, 0, 7, 0.5, threshold_avoid_others=3):
            rule2_clusters.append(i)
        
            
    return rule1_clusters, rule2_clusters, similarity_matrix

# def identify_matching_clusters_lyy(candidate_centers, reference_centers):
#     """识别符合规则1和规则2的候选簇"""
#     similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
#     rule1_clusters = []
#     rule2_clusters = []
    
#     for i, sims in enumerate(similarity_matrix):
#         # 规则1: R7 >= 0.5, 其他 < R7/3 (忽略R0)
#         if check_matching_rules(sims, 0, 2, 0.5, threshold_ratio=3):
#             rule1_clusters.append(i)
#         # 规则2: R0 >= 0.4, 其他 < 0.2 (忽略R7) - 仅当不满足规则1时
#         elif check_matching_rules(sims, 2, 0, 0.5, threshold_avoid_others=3):
#             rule2_clusters.append(i)
        
            
#     return rule1_clusters, rule2_clusters, similarity_matrix

# 保存满足匹配规则的细胞--单细胞filtering
def save_matched_cells(
    indices,
    image_paths,
    labels,
    similarity_matrix,
    rule_type,
    save_dirs,
    target_ref_indices,
    rule=None,
    save_rejected=True,       # ← 新增：是否保存被过滤的细胞
    rejected_dir=None,        # ← 新增：被过滤细胞的保存目录
    rejected_l0_dir=None,
    save_rule_copies=True,
):
    """
    保存匹配的细胞图片（增强版）

    新增功能：
    1. 文件名中加入与最相似恶性参考簇的相似度分数
    2. 被细胞级规则过滤掉的细胞可单独保存到 rejected_dir
       （仅当 rule 不为 None 且 save_rejected=True 且 rejected_dir 不为 None 时生效）
       被过滤细胞的文件名格式：
           rejected_<过滤层>_<rule_type>_cluster<label>_ref<ref_idx>_sim<score>_<原文件名>
    """
    count = 0
    processed_indices = []

    for idx in indices:
        cell_sim_vec = similarity_matrix[idx]

        # ── Layer 0：argmax 检查（检查的是所有参考恶性簇，不是当前 rule 的 target。） ─────────────────────────────────────────
        most_similar_ref = int(np.argmax(cell_sim_vec))
        if most_similar_ref not in target_ref_indices:
            # argmax 不在恶性簇，保存到 rejected（标记 L0）
            if save_rejected and rejected_l0_dir is not None:
                _save_rejected_cell(
                    image_paths[idx], labels[idx], most_similar_ref,
                    cell_sim_vec[most_similar_ref],
                    rule_type, rejected_l0_dir, layer="L0"
                )
            continue

        # ── 找最相似恶性簇及其分数（用于文件命名）──────────────────────
        mal_sims     = {i: cell_sim_vec[i] for i in target_ref_indices}
        best_ref_idx = max(mal_sims, key=mal_sims.get)
        best_sim     = mal_sims[best_ref_idx]

        # ── Layer 1：细胞级规则检查（可选）─────────────────────────────
        if rule is not None:
            passed = check_matching_rules(
                cell_sim_vec,
                ref_idx_target         = rule["target"],
                ref_idx_avoid          = rule["avoid"],
                threshold_target       = rule["threshold_target"],
                threshold_avoid_others = rule.get("threshold_avoid_others"),
                threshold_ratio        = rule.get("threshold_ratio"),
            )
            if not passed:
                # 通过了 L0 但未通过 L1，保存到 rejected（标记 L1）
                if save_rejected and rejected_dir is not None:
                    _save_rejected_cell(
                        image_paths[idx], labels[idx], best_ref_idx,
                        best_sim,
                        rule_type, rejected_dir, layer="L1"
                    )
                continue

        # ── 通过所有检查 → 保存，文件名含相似度分数 ────────────────────
        src_path      = image_paths[idx]
        filename      = os.path.basename(src_path)
        cluster_label = labels[idx]

        prefix   = f"{rule_type}_cluster{cluster_label}_ref{best_ref_idx}_sim{best_sim:.4f}"
        new_name = f"{prefix}_{filename}"

        if save_rule_copies:
            shutil.copy(src_path, os.path.join(save_dirs['rule_dir'], new_name))
        shutil.copy(src_path, os.path.join(save_dirs['total_dir'], new_name))

        count += 1
        processed_indices.append(idx)

    return count, processed_indices


def _save_rejected_cell(src_path, cluster_label, ref_idx, sim_score,
                        rule_type, rejected_dir, layer="L1"):
    """将被过滤的细胞保存到 rejected_dir，文件名含过滤层、规则、相似度分数。"""
    filename = os.path.basename(src_path)
    prefix   = f"rejected_{layer}_{rule_type}_cluster{cluster_label}_ref{ref_idx}_sim{sim_score:.4f}"
    new_name = f"{prefix}_{filename}"
    os.makedirs(rejected_dir, exist_ok=True)
    shutil.copy(src_path, os.path.join(rejected_dir, new_name))
