#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行top4模型的pipeline脚本
该脚本将：
1. 从evaluation_results中读取top4模型
2. 从每个模型的详细结果中找到test_mcc最高的折数
3. 使用对应的模型权重运行pipeline并保存可视化结果
"""
# import os

# # 设置工作目录到项目根目录
# project_root = r'i:\Esemble_Learning\model1\code'  # 替换成您的实际路径
# os.chdir(project_root)
# print(f"已切换到工作目录: {os.getcwd()}")

import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from cross_cluster_matching.models.models import get_model_and_transform
from cross_cluster_matching.data.data_utils import collect_reference_data, collect_image_paths, CellImageDataset
from cross_cluster_matching.models.feature_extractor import extract_features_with_cell_size
from cross_cluster_matching.clustering.K_optimizer import optimize_k_for_candidate_clustering
from cross_cluster_matching.clustering.calculate_consensus_score import calculate_consensus_score
from cross_cluster_matching.clustering.matching_rule_application import identify_matching_clusters, save_matched_cells, check_matching_rules
from cross_cluster_matching.visualization.visualization import (
    count_non_background_pixels,
    visualize_all_clusters_without_images, 
    improved_visualize_clusters_with_images, 
    visualize_matching_clusters,
    analyze_cluster_composition,
    plot_similarity_heatmap,
    visualize_filtered_clusters
)

seed=42
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed_all(seed) #为PyTorch的所有GPU设备设置随机种子
np.random.seed(seed)  # 为NumPy的随机数生成器设置种子
torch.backends.cudnn.deterministic = True #让cuDNN（CUDA深度神经网络库）使用确定性算法
torch.backends.cudnn.benchmark = False # 关闭cuDNN的自动调优功能，确保一致性


def find_best_model_for_top4(
    models_folder,
    evaluation_folder
):
    """
    找到top4模型中每种模型的最佳折数
    """
    # 读取top4模型摘要文件
    top4_summary_path = os.path.join(evaluation_folder, 'top4_models_summary.csv')
    if not os.path.exists(top4_summary_path):
        raise FileNotFoundError(f"找不到top4模型摘要文件: {top4_summary_path}")
    
    top4_df = pd.read_csv(top4_summary_path)
    top4_models = top4_df['model'].tolist()[:4]  # 获取前4个模型
    
    print(f"Top4模型: {top4_models}")
    
    best_model_paths = {}
    
    for model_name in top4_models:
        # 构建模型详细结果文件路径
        model_eval_folder = os.path.join(evaluation_folder, model_name)
        detailed_results_file = f"{model_name}_detailed_results.csv"
        detailed_results_path = os.path.join(model_eval_folder, detailed_results_file)
        
        if not os.path.exists(detailed_results_path):
            print(f"警告: 找不到详细结果文件 {detailed_results_path}，跳过模型 {model_name}")
            continue
        
        # 读取详细结果文件
        detailed_df = pd.read_csv(detailed_results_path)
        
        # 找到test_mcc值最大的折数
        max_mcc_idx = detailed_df['test_mcc'].idxmax()
        best_fold = detailed_df.loc[max_mcc_idx, 'fold']
        
        # 构建模型权重文件路径
        model_filename = f"{model_name.lower()}_fold_{best_fold}.pth"
        model_path = os.path.join(models_folder, model_name, model_filename)
        
        if os.path.exists(model_path):
            best_model_paths[model_name] = {
                'path': model_path,
                'fold': best_fold,
                'mcc': detailed_df.loc[max_mcc_idx, 'test_mcc']
            }
            print(f"模型 {model_name}: 最佳折数 {best_fold}, MCC值 {detailed_df.loc[max_mcc_idx, 'test_mcc']:.4f}, 路径: {model_path}")
        else:
            print(f"警告: 找不到模型权重文件 {model_path}")
    
    return best_model_paths

def run_pipeline_for_model(
    model_name,
    model_path,
    malignant_cells_dir,
    benign_cells_dir,
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
    为单个模型运行pipeline并保存结果
    """
    print(f"开始运行模型 {model_name} 的pipeline...")
    
    # 尝试从对应的日志文件中读取 mean 和 std
    log_file_path = model_path.replace('.pth', '.log')
    mean = [0.485, 0.456, 0.406]  # 默认ImageNet均值
    std = [0.229, 0.224, 0.225]   # 默认ImageNet标准差
    
    # 如果存在日志文件，尝试从中读取
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if '均值:' in line or 'Mean:' in line and i + 1 < len(lines):
                        mean_line = lines[i + 1].strip()
                        if '[' in mean_line and ']' in mean_line:
                            values_str = mean_line[mean_line.find('[')+1:mean_line.find(']')]
                            mean = [float(x.strip()) for x in values_str.split(',')]
                    elif '标准差:' in line or 'Std:' in line and i + 1 < len(lines):
                        std_line = lines[i + 1].strip()
                        if '[' in std_line and ']' in std_line:
                            values_str = std_line[std_line.find('[')+1:std_line.find(']')]
                            std = [float(x.strip()) for x in values_str.split(',')]
        except:
            print(f"读取日志文件 {log_file_path} 失败，使用默认均值和标准差")
    
    try:
        # 1. 初始化环境与模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, transform, _ = get_model_and_transform(model_name, device, model_path, mean= mean, std = std)
        
        # 2. 处理参考域 (Reference Domain)
        print("======= 处理参考细胞 =======")
        ref_paths, ref_sources = collect_reference_data(malignant_cells_dir, benign_cells_dir)
        
        # 计算大小 (逻辑保持不变，但代码更紧凑)
        ref_sizes = np.log1p([count_non_background_pixels(p) for p in tqdm(ref_paths, desc="Ref Sizes")])
        
        # 提取特征
        ref_loader = DataLoader(
            CellImageDataset(ref_paths, transform=transform, source_labels=ref_sources),
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        ref_feats, ref_sources, ref_paths = extract_features_with_cell_size(
            model, ref_loader, device, ref_sizes, feature_layer, cell_size_weight
        )
        
        # Reference PCA & Clustering
        pca = PCA(n_components=min(pca_dim, ref_feats.shape[0], ref_feats.shape[1]))
        ref_norm = normalize(pca.fit_transform(ref_feats))
        
        kmeans = KMeans(n_clusters=reference_k, random_state=42).fit(ref_norm)
        ref_results = {
            'features': ref_norm, 'labels': kmeans.labels_, 'source_labels': ref_sources,
            'image_paths': ref_paths, 'cluster_centers': kmeans.cluster_centers_,
            'n_clusters': reference_k, 'pca': pca
        }
        
        # Reference Visualization 
        ref_vis_dir = os.path.join(output_base_dir, model_name, f'reference_visualization_pca{pca_dim}')
        os.makedirs(ref_vis_dir, exist_ok=True)


        reference_umap = visualize_all_clusters_without_images(
            ref_results['features'], 
            ref_results['labels'], 
            ref_results['source_labels'], 
            "Reference", 
            ref_vis_dir
        )

        improved_visualize_clusters_with_images(
            ref_results['features'], 
            ref_results['labels'], 
            ref_results['source_labels'], 
            ref_results['image_paths'],
            "Reference", 
            ref_vis_dir,
            ref_sizes
        )
        analyze_cluster_composition(
            ref_results['labels'], 
            ref_results['source_labels'],
            ref_results['image_paths'],
            "Reference", 
            ref_vis_dir
            )
        
        if save_cluster_images:
            # 类型名称映射
            category_names = {
                0: "Candidate",
                1: "LUAD",
                2: "LUSC",
                3: "SCLC",
                4: "Benign",
                5: "Malignant"
            }
            
            # 创建基础目录
            clusters_images_dir = os.path.join(ref_vis_dir, "cluster_images")
            os.makedirs(clusters_images_dir, exist_ok=True)
            
            # 为每个簇创建目录
            unique_clusters = sorted(set(ref_results['labels']))
            for cluster in unique_clusters:
                cluster_dir = os.path.join(clusters_images_dir, f"cluster_{cluster}")
                os.makedirs(cluster_dir, exist_ok=True)
                
                # 为每种类型创建子目录
                for source_id, category in category_names.items():
                    category_dir = os.path.join(cluster_dir, category)
                    os.makedirs(category_dir, exist_ok=True)
            
            # 复制图像到对应目录
            import shutil
            from pathlib import Path
            
            print(f"正在将图像复制到各个簇文件夹...")
            for img_path, label, source in tqdm(zip(ref_results['image_paths'], 
                                                   ref_results['labels'], 
                                                   ref_results['source_labels']),
                                               total=len(ref_results['image_paths']),
                                               desc="复制图像"):
                # 获取目标目录
                cluster_num = label
                category = category_names[source]
                target_dir = os.path.join(clusters_images_dir, f"cluster_{cluster_num}", category)
                
                # 获取文件名并复制
                file_name = Path(img_path).name
                target_path = os.path.join(target_dir, file_name)
                
                # 复制文件
                shutil.copy2(img_path, target_path)
        return True
        
    except Exception as e:
        print(f"模型 {model_name} pipeline执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("运行Top4模型Pipeline脚本")
    print("="*50)
    
    # 配置路径
    models_folder = r"H:\limr\project\urine\result\classification_model\UR_SingleCell\train_val_models"
    evaluation_folder = r"H:\limr\project\urine\result\classification_model\UR_SingleCell\evaluation_results"
    
    # 结果保存路径 - 请在这里填写您的结果保存路径
    output_base_dir = r"H:\limr\project\urine\result\classification_model\UR_SingleCell\Top4_clustering_analysis"  # 请修改为您的实际路径
    
    # 细胞数据路径
    malignant_cells_dir = r"H:\limr\project\urine\rawdata\classification_model\UR_SingleCell\malignant"
    benign_cells_dir = r"H:\limr\project\urine\rawdata\classification_model\UR_SingleCell\benign"
    
    # 检查路径是否存在
    if not os.path.exists(models_folder):
        print(f"错误: 模型文件夹不存在: {models_folder}")
        return
    
    if not os.path.exists(evaluation_folder):
        print(f"错误: 评估结果文件夹不存在: {evaluation_folder}")
        return
    
    if not os.path.exists(malignant_cells_dir):
        print(f"错误: 恶性细胞文件夹不存在: {malignant_cells_dir}")
        return
    
    if not os.path.exists(benign_cells_dir):
        print(f"错误: 良性细胞文件夹不存在: {benign_cells_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"模型文件夹: {models_folder}")
    print(f"评估结果文件夹: {evaluation_folder}")
    print(f"输出基础目录: {output_base_dir}")
    print()
    
    # 1. 找到top4模型中每种模型的最佳折数
    print("步骤1: 查找Top4模型的最佳折数...")
    best_model_paths = find_best_model_for_top4(models_folder, evaluation_folder)
    
    if not best_model_paths:
        print("没有找到任何有效的模型权重文件")
        return
    
    print(f"找到 {len(best_model_paths)} 个有效模型")
    print()
    
    # 2. 为每个最佳模型运行pipeline
    print("步骤2: 运行每个最佳模型的pipeline...")
    successful_models = []
    failed_models = []
    
    for model_name, model_info in best_model_paths.items():
        print(f"\n{'='*60}")
        print(f"运行模型: {model_name} (折数: {model_info['fold']}, MCC: {model_info['mcc']:.4f})")
        print(f"权重路径: {model_info['path']}")
        print(f"{'='*60}")
        
        success = run_pipeline_for_model(
            model_name=model_name,
            model_path=model_info['path'],
            malignant_cells_dir=malignant_cells_dir,
            benign_cells_dir=benign_cells_dir,
            output_base_dir=output_base_dir,
            batch_size=8,
            num_workers=0,
            cell_size_weight=1.0,
            feature_layer='penultimate',
            pca_dim=32,
            reference_k=10
        )
        
        if success:
            successful_models.append(model_name)
            print(f"✓ {model_name} pipeline执行成功!")
        else:
            failed_models.append(model_name)
            print(f"✗ {model_name} pipeline执行失败!")
    
    # 输出最终结果
    print(f"\n{'='*60}")
    print("执行完成总结:")
    print(f"总共处理: {len(best_model_paths)} 个模型")
    print(f"成功: {len(successful_models)} 个")
    print(f"失败: {len(failed_models)} 个")
    print("="*60)
    
    if successful_models:
        print("\n成功的模型:")
        for model in successful_models:
            print(f"  ✓ {model}")
    
    if failed_models:
        print("\n失败的模型:")
        for model in failed_models:
            print(f"  ✗ {model}")

if __name__ == "__main__":
    main()