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
from tqdm import tqdm

from cross_cluster_matching.models.models import get_model_and_transform
from cross_cluster_matching.data.data_utils import collect_reference_data, collect_image_paths, CellImageDataset, extract_patient_id
from cross_cluster_matching.models.feature_extractor import extract_features_with_cell_size
from cross_cluster_matching.clustering.K_optimizer import optimize_k_for_candidate_clustering
from cross_cluster_matching.clustering.calculate_consensus_score import calculate_consensus_score
from cross_cluster_matching.clustering.matching_rule_application import identify_matching_clusters_urine, save_matched_cells, check_matching_rules
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


def run_pipeline_urine(
    positive_folder_path, negative_folder_path, malignant_cells_dir, benign_cells_dir,
    model_path, base_save_dir, 
    feature_layer='penultimate', reference_k=10, max_candidate_k=20, 
    batch_size=16, num_workers=0, cell_size_weight=1.0, model_type='ResNeXt', pca_dim=32 , mean = None , std = None, malignant_ref_clusters=None
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
    """
    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 1. 初始化环境与模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, _ = get_model_and_transform(model_type, device, model_path, mean= mean, std = std)
    
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
    ref_vis_dir = os.path.join(base_save_dir, f'reference_visualization_pca{pca_dim}')
    os.makedirs(ref_vis_dir, exist_ok=True)


    reference_umap = visualize_all_clusters_without_images(
        ref_results['features'], 
        ref_results['labels'], 
        ref_results['source_labels'], 
        "Reference", 
        ref_vis_dir
    )

    reference_umap_1 = visualize_all_clusters_without_images_1(
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

    # 3. 准备病理片子列表
    patient_folders = []
    patient_folders.extend([(entry.path, 'positive') for entry in os.scandir(positive_folder_path) if entry.is_dir()])
    patient_folders.extend([(entry.path, 'negative') for entry in os.scandir(negative_folder_path) if entry.is_dir()])
    
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
        
        # === 分支逻辑 ===
        matched_count_r1 = 0
        matched_count_r2 = 0
        matched_count_r3 = 0
        
        # 创建保存目录
        dirs = {
            'rule1': os.path.join(save_dir, 'rule1_matched_cells'),
            'rule2': os.path.join(save_dir, 'rule2_matched_cells'),
            'rule3': os.path.join(save_dir, 'rule3_matched_cells'),
            'total': os.path.join(save_dir, 'matched_malignant_cells')
        }
        for d in dirs.values(): os.makedirs(d, exist_ok=True)
        
        if len(cand_norm) >= 32:
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

            # 识别匹配簇
            r1_clusters, r2_clusters, r3_clusters, similarity_matrix = identify_matching_clusters_urine(
                cand_results['cluster_centers'], ref_results['cluster_centers']
            )
            
            # 计算细胞级别的相似度用于保存
            cell_sims = cosine_similarity(cand_results['features'], ref_results['cluster_centers'])
            
            # 保存 Rule 1
            indices_r1 = [i for i, l in enumerate(cand_results['labels']) if l in r1_clusters]
            c1, _ = save_matched_cells(indices_r1, cand_paths, cand_results['labels'], cell_sims, 'rule1', 
                                     {'rule_dir': dirs['rule1'], 'total_dir': dirs['total']}, malignant_ref_clusters)
            
            # 保存 Rule 2 (排除已处理的)
            indices_r2 = [i for i, l in enumerate(cand_results['labels']) if l in r2_clusters]
            c2, _ = save_matched_cells(indices_r2, cand_paths, cand_results['labels'], cell_sims, 'rule2', 
                                     {'rule_dir': dirs['rule2'], 'total_dir': dirs['total']}, malignant_ref_clusters)
            
            # 保存 Rule 3 (排除已处理的)
            indices_r3 = [i for i, l in enumerate(cand_results['labels']) if l in r3_clusters]
            c3, _ = save_matched_cells(indices_r3, cand_paths, cand_results['labels'], cell_sims, 'rule3', 
                                     {'rule_dir': dirs['rule3'], 'total_dir': dirs['total']}, malignant_ref_clusters)
            
            matched_count_r1, matched_count_r2, matched_count_r3 = c1, c2, c3
            # 生成matching_pairs参数
            # 为R0和R5创建匹配对，包括相似度信息
            matching_pairs = []
            
            # 添加Rule1匹配的簇（与R5匹配）
            for cand_idx in r1_clusters:
                matching_pairs.append((malignant_ref_clusters[0], cand_idx, similarity_matrix[cand_idx, malignant_ref_clusters[0]]))
            
            # 添加Rule2匹配的簇（与R0匹配）
            for cand_idx in r2_clusters:
                matching_pairs.append((malignant_ref_clusters[1], cand_idx, similarity_matrix[cand_idx, malignant_ref_clusters[1]]))

            # 添加Rule3匹配的簇（与R9匹配）
            for cand_idx in r3_clusters:
                matching_pairs.append((malignant_ref_clusters[2], cand_idx, similarity_matrix[cand_idx, malignant_ref_clusters[2]]))
    
            calculate_consensus_score(
                            ref_results, 
                            cand_results, 
                            matching_pairs, 
                            save_dir
                        )
            rule_matched_clusters = {
                'rule1': r1_clusters,
                'rule2': r2_clusters,
                'rule3': r3_clusters                
            }

            visualize_matching_clusters(
                ref_results,
                cand_results,
                matching_pairs,
                save_dir,
                reference_umap,
                rule_matched_clusters
            )
            
            visualize_all_clusters_without_images(
                cand_results['features'], 
                cand_results['labels'], 
                cand_results['source_labels'], 
                pid, 
                save_dir,
                reference_umap
            )
            improved_visualize_clusters_with_images(
                cand_results['features'], 
                cand_results['labels'], 
                cand_results['source_labels'], 
                cand_paths,
                pid, 
                save_dir,
                cand_sizes
            )
            visualize_filtered_clusters(
                ref_results, 
                cand_results, 
                matching_pairs,
                save_dir,
                rule_matched_clusters= rule_matched_clusters,
                target_ref_indices= malignant_ref_clusters                
            )
            xlabel = [f"C{i}" for i in range(cand_results['n_clusters'])]
            ylabel = [f"R{i}" for i in range(ref_results['n_clusters'])]
            plot_similarity_heatmap(similarity_matrix.T, xlabel, ylabel, f"{pid} Similarity Matrix", os.path.join(save_dir, 'similarity_heatmap.png'))
            
        else:
            # B. 单细胞模式
            cell_sims = cosine_similarity(cand_norm, ref_results['cluster_centers'])
            
            # 筛选 Rule 1
            indices_r1 = [i for i, s in enumerate(cell_sims) if check_matching_rules(s, 0, [4, 9], 0.5, threshold_ratio=3)]
            c1, processed_r1 = save_matched_cells(indices_r1, cand_paths, [0]*len(cand_paths), cell_sims, 'rule1',
                                                {'rule_dir': dirs['rule1'], 'total_dir': dirs['total']}, [0, 4, 9])
            
            # 筛选 Rule 2 (排除 Rule 1)
            indices_r2 = [i for i, s in enumerate(cell_sims) 
                          if i not in processed_r1 and check_matching_rules(s, 4, [0, 9], 0.5, threshold_ratio=3)]
            c2, processed_r2 = save_matched_cells(indices_r2, cand_paths, [0]*len(cand_paths), cell_sims, 'rule2',
                                     {'rule_dir': dirs['rule2'], 'total_dir': dirs['total']}, [0, 4, 9])
            
            # 筛选 Rule 3 (排除 Rule 2)
            indices_r3 = [i for i, s in enumerate(cell_sims) 
                          if i not in processed_r2 and check_matching_rules(s, 9, [0, 4], 0.5, threshold_ratio=3)]
            c3, _ = save_matched_cells(indices_r3, cand_paths, [0]*len(cand_paths), cell_sims, 'rule3',
                                     {'rule_dir': dirs['rule3'], 'total_dir': dirs['total']}, [0, 4, 9])
            
            matched_count_r1, matched_count_r2, matched_count_r3 = c1, c2, c3

            xlabel = [f"Cell{i}" for i in range(len(cell_sims))]
            ylabel = [f"R{i}" for i in range(ref_results['n_clusters'])]
            plot_similarity_heatmap(cell_sims.T, xlabel, ylabel, f"Patient {pid} Cell Similarity Matrix", os.path.join(save_dir, 'Cell_similarity_heatmap.png'))


        # 记录统计信息
        results_stats.append({
            'Patient_ID': pid,
            'Status': status,
            'Rule1': matched_count_r1,
            'Rule2': matched_count_r2,
            'Rule3': matched_count_r3,
            'Total': matched_count_r1 + matched_count_r2 + matched_count_r3
        })
        
        print(f"Patient {pid} Done. Matched: {matched_count_r1 + matched_count_r2 + matched_count_r3}")

    # 保存统计结果
    pd.DataFrame(results_stats).to_excel(os.path.join(base_save_dir, 'stats.xlsx'))
