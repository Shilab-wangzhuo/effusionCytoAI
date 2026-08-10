#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行top4模型的pipeline脚本（多癌种版本）

✅ 修改说明（相对于原二分类版本）：
┌──────┬──────────────────────────────────┬────────────────────────────────────────────────┐
│ 改动 │ 位置                             │ 说明                                           │
├──────┼──────────────────────────────────┼────────────────────────────────────────────────┤
│  1   │ run_pipeline_for_model 参数      │ malignant/benign两个路径 → cancer_dirs字典     │
│  2   │ collect_reference_data 调用      │ 传2个固定路径 → 传cancer_dirs字典              │
│  3   │ category_names 映射              │ 硬编码 → 动态从cancer_dirs.keys()生成          │
│  4   │ main() 路径配置                  │ 2个独立变量 → cancer_dirs字典                  │
│  5   │ 路径存在性检查                   │ 硬编码检查2个路径 → 循环检查所有cancer_dirs    │
│  6   │ run_pipeline_for_model 调用处    │ 传malignant/benign两参数 → 传cancer_dirs       │
│  7   │ 簇子目录创建与图像复制           │ 遍历固定category_names → 动态遍历类别名        │
│  8   │ build_category_config 调用       │ 新增 include_candidate=False，避免插入Candidate│
│  9   │ improved_visualize调用           │ log_sizes 改为关键字参数，避免位置参数冲突     │
└──────┴──────────────────────────────────┴────────────────────────────────────────────────┘
"""

import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from shilab_cancer_classifier.clustering.model import get_model_and_transform
from cross_cluster_matching.data.data_utils import collect_image_paths, CellImageDataset
from cross_cluster_matching.models.feature_extractor import extract_features_with_cell_size
from cross_cluster_matching.clustering.K_optimizer import optimize_k_for_candidate_clustering
from cross_cluster_matching.clustering.calculate_consensus_score import calculate_consensus_score
from cross_cluster_matching.clustering.matching_rule_application import (
    identify_matching_clusters, save_matched_cells, check_matching_rules
)
from shilab_cancer_classifier.clustering.visualization import (
    count_non_background_pixels,
    visualize_all_clusters_without_images,
    visualize_all_clusters_without_images_1,
    improved_visualize_clusters_with_images,
    visualize_matching_clusters,
    analyze_cluster_composition,
    plot_similarity_heatmap,
    visualize_filtered_clusters,
    build_category_config
)


# ============================================================
# 工具函数
# ============================================================

def collect_reference_data(cancer_dirs: dict):
    """
    收集所有癌种的图像路径和对应的 source_label。

    参数:
        cancer_dirs (dict):
            有序字典，键为癌种名称，值为图像目录路径。
            例如:
                {
                    'Gastrointestinal_Breast': r'J:\...\Gastrointestinal_Breast',
                    'Gynecologic'            : r'J:\...\Gynecologic',
                    'Lung'                   : r'J:\...\Lung',
                    'Mesothelioma'           : r'J:\...\Mesothelioma',
                }

    返回:
        all_paths   (list)       : 所有图像的绝对路径
        all_sources (np.ndarray) : 与 all_paths 等长，值为类别索引 (0, 1, 2, ...)
                                   0 -> 第1个癌种，1 -> 第2个癌种，以此类推
    """
    VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_paths   = []
    all_sources = []

    for source_id, (cancer_name, folder) in enumerate(cancer_dirs.items()):
        paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(VALID_EXTS)
        ])
        all_paths.extend(paths)
        all_sources.extend([source_id] * len(paths))
        print(f"  [{source_id}] {cancer_name}: {len(paths)} 张图像")

    return all_paths, np.array(all_sources)


def find_best_model_for_top4(models_folder, evaluation_folder):
    """
    从 all_models_summary.csv 中读取前4个模型，
    对每个模型查找 test_mcc 最高的折数，
    并将结果保存为 top4_models_summary.csv。

    返回:
        best_model_paths (dict):
            {
                'ModelName': {
                    'path': '/path/to/model.pth',
                    'fold': 3,
                    'mcc':  0.8765
                },
                ...
            }
    """
    all_summary_path = os.path.join(evaluation_folder, 'all_models_summary.csv')
    if not os.path.exists(all_summary_path):
        raise FileNotFoundError(f"找不到模型汇总文件: {all_summary_path}")

    all_df = pd.read_csv(all_summary_path)

    if 'model' not in all_df.columns:
        raise ValueError(f"{all_summary_path} 中缺少 'model' 列")
    if all_df.empty:
        raise ValueError(f"{all_summary_path} 为空，没有模型信息")

    top4_models = all_df['model'].dropna().tolist()[:4]
    if not top4_models:
        raise ValueError("all_models_summary.csv 中没有可用的模型")

    print(f"从 all_models_summary.csv 中读取到前4个模型: {top4_models}")

    best_model_paths  = {}
    top4_summary_rows = []

    for rank, model_name in enumerate(top4_models, start=1):
        model_eval_folder    = os.path.join(evaluation_folder, model_name)
        detailed_results_path = os.path.join(
            model_eval_folder, f"{model_name}_detailed_results.csv"
        )

        if not os.path.exists(detailed_results_path):
            print(f"警告: 找不到详细结果文件 {detailed_results_path}，跳过模型 {model_name}")
            continue

        detailed_df = pd.read_csv(detailed_results_path)

        for col in ['test_mcc', 'fold']:
            if col not in detailed_df.columns:
                print(f"警告: {detailed_results_path} 中缺少 '{col}' 列，跳过模型 {model_name}")
                break
        else:
            if detailed_df['test_mcc'].isna().all():
                print(f"警告: {detailed_results_path} 的 test_mcc 全为空，跳过模型 {model_name}")
                continue

            max_mcc_idx = detailed_df['test_mcc'].idxmax()
            best_row    = detailed_df.loc[max_mcc_idx]
            best_fold   = best_row['fold']
            best_mcc    = best_row['test_mcc']

            model_filename = f"{model_name.lower()}_fold_{best_fold}.pth"
            model_path     = os.path.join(models_folder, model_name, model_filename)

            if not os.path.exists(model_path):
                print(f"警告: 找不到模型权重文件 {model_path}，跳过模型 {model_name}")
                continue

            best_model_paths[model_name] = {
                'path': model_path,
                'fold': best_fold,
                'mcc':  best_mcc
            }

            summary_row = {
                'rank'         : rank,
                'model'        : model_name,
                'best_fold'    : best_fold,
                'best_test_mcc': best_mcc,
                'model_path'   : model_path
            }
            for col in ['test_accuracy', 'test_balanced_accuracy',
                        'test_f1_macro', 'test_f1_weighted',
                        'test_auc_macro_ovr', 'test_auc_weighted_ovr', 'test_mcc']:
                if col in detailed_df.columns:
                    summary_row[col] = best_row[col]

            top4_summary_rows.append(summary_row)
            print(f"模型 {model_name}: 最佳折数 {best_fold}, MCC {best_mcc:.4f}, 路径: {model_path}")

    if top4_summary_rows:
        top4_df   = pd.DataFrame(top4_summary_rows)
        save_path = os.path.join(evaluation_folder, 'top4_models_summary.csv')
        top4_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n已保存 top4 模型最佳折汇总文件: {save_path}")
    else:
        print("\n警告: 没有可保存的 top4 模型汇总信息")

    return best_model_paths


# ============================================================
# 核心 Pipeline
# ============================================================

def run_pipeline_for_model(
    model_name,
    model_path,
    cancer_dirs,
    output_base_dir,
    batch_size=8,
    num_workers=0,
    cell_size_weight=1.0,
    feature_layer='penultimate',
    pca_dim=32,
    reference_k=10,
    save_cluster_images=True
):
    """
    为单个模型运行 pipeline 并保存结果。

    参数:
        model_name      : 模型名称（字符串）
        model_path      : 模型权重 .pth 文件路径
        cancer_dirs     : 有序字典，键为癌种名，值为图像目录路径。
                          例如:
                              {
                                  'Gastrointestinal_Breast': r'J:\...\Gastrointestinal_Breast',
                                  'Gynecologic'            : r'J:\...\Gynecologic',
                                  'Lung'                   : r'J:\...\Lung',
                                  'Mesothelioma'           : r'J:\...\Mesothelioma',
                              }
        output_base_dir : 输出根目录
        batch_size      : DataLoader batch size
        num_workers     : DataLoader num_workers
        cell_size_weight: 细胞大小权重
        feature_layer   : 特征提取层（'penultimate'）
        pca_dim         : PCA 降维维度
        reference_k     : KMeans 聚类数
        save_cluster_images: 是否将图像按簇复制到对应文件夹
    """
    print(f"开始运行模型 {model_name} 的 pipeline...")

    # 固定随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # 尝试从日志文件读取 mean / std
    log_file_path = model_path.replace('.pth', '.log')
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                if ('均值:' in line or 'Mean:' in line) and '[' in line and ']' in line:
                    vals_str = line[line.find('[')+1 : line.find(']')]
                    mean = [float(x.strip()) for x in vals_str.split(',')]
                    print(f"  从日志读取均值: {mean}")
                elif ('标准差:' in line or 'Std:' in line) and '[' in line and ']' in line:
                    vals_str = line[line.find('[')+1 : line.find(']')]
                    std = [float(x.strip()) for x in vals_str.split(',')]
                    print(f"  从日志读取标准差: {std}")
        except Exception:
            print(f"  读取日志文件 {log_file_path} 失败，使用默认 mean/std")
    else:
        print(f"  日志文件 {log_file_path} 不存在，使用默认 mean/std")

    try:
        # --------------------------------------------------------
        # 1. 初始化模型
        # --------------------------------------------------------
        device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(cancer_dirs)
        print(f"  设备: {device}，类别数: {num_classes}")

        model, transform, _ = get_model_and_transform(
            model_name, device, model_path,
            mean=mean, std=std, num_classes=num_classes
        )

        # --------------------------------------------------------
        # 2. 收集参考域数据
        # --------------------------------------------------------
        print("======= 处理参考细胞 =======")
        ref_paths, ref_sources = collect_reference_data(cancer_dirs)

        # 计算细胞大小（对数变换）
        print("计算细胞大小...")
        ref_sizes = np.log1p([
            count_non_background_pixels(p)
            for p in tqdm(ref_paths, desc="Ref Sizes")
        ])

        # 提取特征
        ref_loader = DataLoader(
            CellImageDataset(ref_paths, transform=transform, source_labels=ref_sources),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        ref_feats, ref_sources, ref_paths = extract_features_with_cell_size(
            model, ref_loader, device, ref_sizes, feature_layer, cell_size_weight
        )

        # PCA 降维 & KMeans 聚类
        pca      = PCA(n_components=min(pca_dim, ref_feats.shape[0], ref_feats.shape[1]))
        ref_norm = normalize(pca.fit_transform(ref_feats))
        kmeans   = KMeans(n_clusters=reference_k, random_state=42).fit(ref_norm)

        ref_results = {
            'features'       : ref_norm,
            'labels'         : kmeans.labels_,
            'source_labels'  : ref_sources,
            'image_paths'    : ref_paths,
            'cluster_centers': kmeans.cluster_centers_,
            'n_clusters'     : reference_k,
            'pca'            : pca
        }

        # --------------------------------------------------------
        # 3. 可视化
        # --------------------------------------------------------
        ref_vis_dir = os.path.join(
            output_base_dir, model_name, f'reference_visualization_pca{pca_dim}'
        )
        os.makedirs(ref_vis_dir, exist_ok=True)

        # ⚠️ 关键：
        # collect_reference_data 生成的 source_label 从 0 开始对应 cancer_dirs 的第一个癌种，
        # 没有 Candidate，所以必须 include_candidate=False。
        # 若使用 include_candidate=True，会插入 Candidate=0，导致所有标签错位。
        category_names, category_colors, category_markers = build_category_config(
            cancer_dirs,
            include_candidate=False   # ← 必须为 False
        )

        # 打印映射关系，便于调试确认
        print("\n当前画图类别映射:")
        for k, v in category_names.items():
            print(f"  {k}: {v}")
        print("当前数据中实际存在的 source_label:")
        print(sorted(set(int(x) for x in ref_results['source_labels'])))
        print()

        # 3-1. 全局 UMAP：颜色=簇，形状=类别
        visualize_all_clusters_without_images(
            ref_results['features'],
            ref_results['labels'],
            ref_results['source_labels'],
            "Reference",
            ref_vis_dir,
            category_names=category_names,
            category_colors=category_colors,
            category_markers=category_markers
        )

        # 3-2. 全局 UMAP：颜色=类别，形状=圆点
        visualize_all_clusters_without_images_1(
            ref_results['features'],
            ref_results['labels'],
            ref_results['source_labels'],
            "Reference",
            ref_vis_dir,
            category_names=category_names,
            category_colors=category_colors
        )

        # 3-3. 每个簇的细胞缩略图 UMAP
        # ⚠️ 注意：log_sizes 必须用关键字参数传入，
        #          不能作为第7个位置参数，否则会与 category_names 冲突
        improved_visualize_clusters_with_images(
            ref_results['features'],
            ref_results['labels'],
            ref_results['source_labels'],
            ref_results['image_paths'],
            "Reference",
            ref_vis_dir,
            category_names=category_names,
            category_colors=category_colors,
            log_sizes=ref_sizes             # ← 关键字参数
        )

        # 3-4. 聚类组成堆叠柱状图（保存在 ref_vis_dir 根目录，不在 cluster_images 子目录）
        analyze_cluster_composition(
            ref_results['labels'],
            ref_results['source_labels'],
            ref_results['image_paths'],
            "Reference",
            ref_vis_dir,
            category_names=category_names,
            category_colors=category_colors
        )

        # --------------------------------------------------------
        # 4. 将各簇图像复制到对应文件夹
        # --------------------------------------------------------
        if save_cluster_images:
            clusters_images_dir = os.path.join(ref_vis_dir, "cluster_images")
            os.makedirs(clusters_images_dir, exist_ok=True)

            unique_clusters = sorted(set(int(x) for x in ref_results['labels']))

            # 创建子目录结构：cluster_X / 癌种名
            for cluster in unique_clusters:
                cluster_dir = os.path.join(clusters_images_dir, f"cluster_{cluster}")
                os.makedirs(cluster_dir, exist_ok=True)
                for category in category_names.values():
                    os.makedirs(os.path.join(cluster_dir, category), exist_ok=True)

            # 复制图像
            print("正在将图像复制到各个簇文件夹...")
            skip_count = 0
            for img_path, label, source in tqdm(
                zip(
                    ref_results['image_paths'],
                    ref_results['labels'],
                    ref_results['source_labels']
                ),
                total=len(ref_results['image_paths']),
                desc="复制图像"
            ):
                source = int(source)
                label  = int(label)

                if source not in category_names:
                    print(f"警告: source_label={source} 不在 category_names 中，跳过: {img_path}")
                    skip_count += 1
                    continue

                category   = category_names[source]
                target_dir = os.path.join(clusters_images_dir, f"cluster_{label}", category)
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(img_path, os.path.join(target_dir, Path(img_path).name))

            if skip_count > 0:
                print(f"警告: 共有 {skip_count} 张图像因 source_label 不匹配而被跳过")

            print(f"图像已按簇和类型分类保存到: {clusters_images_dir}")

        return True

    except Exception as e:
        print(f"模型 {model_name} pipeline 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 主函数
# ============================================================

def main():
    print("运行 Top4 模型 Pipeline 脚本（多癌种版本）")
    print("=" * 60)

    # ----------------------------------------------------------
    # 路径配置
    # ----------------------------------------------------------
    cancer_dirs = {
        'Gastrointestinal_Breast': r"J:\limr\effusion\rawdata\cancer_classification_6.0\Gastrointestinal_Breast",
        'Gynecologic'            : r"J:\limr\effusion\rawdata\cancer_classification_6.0\Gynecologic",
        'Lung'                   : r"J:\limr\effusion\rawdata\cancer_classification_6.0\Lung",
        'Mesothelioma'           : r"J:\limr\effusion\rawdata\cancer_classification_6.0\Mesothelioma",
    }

    models_folder     = r"J:\limr\effusion\result\cancer_classification\exp_0621\train_val_models"
    evaluation_folder = r"J:\limr\effusion\result\cancer_classification\exp_0621\evaluation_results"
    output_base_dir   = r"J:\limr\effusion\result\cancer_classification\exp_0621\Top4_clustering_analysis"

    # ----------------------------------------------------------
    # 路径检查
    # ----------------------------------------------------------
    for path_name, path_value in [
        ('模型文件夹',     models_folder),
        ('评估结果文件夹', evaluation_folder),
    ]:
        if not os.path.exists(path_value):
            print(f"错误: {path_name} 不存在: {path_value}")
            return

    for cancer_name, cancer_path in cancer_dirs.items():
        if not os.path.exists(cancer_path):
            print(f"错误: 癌种 [{cancer_name}] 的图像目录不存在: {cancer_path}")
            return

    os.makedirs(output_base_dir, exist_ok=True)

    print(f"模型文件夹:     {models_folder}")
    print(f"评估结果文件夹: {evaluation_folder}")
    print(f"输出基础目录:   {output_base_dir}")
    print(f"癌种类别数:     {len(cancer_dirs)}  →  {list(cancer_dirs.keys())}")
    print()

    # ----------------------------------------------------------
    # 步骤1：找到 top4 模型的最佳折数
    # ----------------------------------------------------------
    print("步骤1: 查找 Top4 模型的最佳折数...")
    best_model_paths = find_best_model_for_top4(models_folder, evaluation_folder)

    if not best_model_paths:
        print("没有找到任何有效的模型权重文件")
        return

    print(f"找到 {len(best_model_paths)} 个有效模型\n")

    # ----------------------------------------------------------
    # 步骤2：为每个最佳模型运行 pipeline
    # ----------------------------------------------------------
    print("步骤2: 运行每个最佳模型的 pipeline...")
    successful_models = []
    failed_models     = []

    for model_name, model_info in best_model_paths.items():
        print(f"\n{'='*60}")
        print(f"运行模型: {model_name}  折数: {model_info['fold']}  MCC: {model_info['mcc']:.4f}")
        print(f"权重路径: {model_info['path']}")
        print(f"{'='*60}")

        success = run_pipeline_for_model(
            model_name          = model_name,
            model_path          = model_info['path'],
            cancer_dirs         = cancer_dirs,
            output_base_dir     = output_base_dir,
            batch_size          = 8,
            num_workers         = 0,
            cell_size_weight    = 1.0,
            feature_layer       = 'penultimate',
            pca_dim             = 32,
            reference_k         = 10,
            save_cluster_images = True
        )

        if success:
            successful_models.append(model_name)
            print(f"✓ {model_name} pipeline 执行成功!")
        else:
            failed_models.append(model_name)
            print(f"✗ {model_name} pipeline 执行失败!")

    # ----------------------------------------------------------
    # 输出汇总
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("执行完成总结:")
    print(f"总共处理: {len(best_model_paths)} 个模型")
    print(f"成功:     {len(successful_models)} 个")
    print(f"失败:     {len(failed_models)} 个")
    print("=" * 60)
    if successful_models:
        print("\n成功的模型:")
        for m in successful_models:
            print(f"  ✓ {m}")
    if failed_models:
        print("\n失败的模型:")
        for m in failed_models:
            print(f"  ✗ {m}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"开始时间: {current_time}")
    main()
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"结束时间: {end_time}")
