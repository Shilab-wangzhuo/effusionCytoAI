# clustering/K_optimizer.py
# 用来存放K值优化相关函数

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import shutil


# 画K值优化曲线
def plot_k_optimization_curve(k_values, scores, best_k, patient_id, save_dir):
    """
    绘制K值优化曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, 'o-', linewidth=2, markersize=8)
    
    # 标记最佳K值
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    
    plt.xlabel('k (Number of Clusters)', fontsize=24, fontweight='bold', labelpad = 20)
    plt.ylabel('Weighted Score', fontsize=24, fontweight='bold', labelpad = 20)
    plt.title(f'{patient_id} K-Optimization', fontsize=28, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16)
    plt.margins(0.1)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    save_path = os.path.join(save_dir, 'k_optimization_plot.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

# 优化candidate domain的聚类簇数
def optimize_k_for_candidate_clustering(
    patient_candidate_normalized, 
    patient_candidate_source_labels, 
    patient_candidate_image_paths,
    reference_results,
    patient_save_dir,
    patient_id,
    max_k=20,
    malignant_ref_clusters=None
):
    """
    优化候选域聚类的k值
    
    参数:
    patient_candidate_normalized - 归一化后的候选域特征
    patient_candidate_source_labels - 候选域样本的来源标签
    patient_candidate_image_paths - 候选域样本的图像路径
    reference_results - 参考域聚类结果字典
    patient_save_dir - 保存结果的目录
    patient_id - 病理片子ID
    candidate_k - 默认候选k值
    max_k - 最大尝试的k值
    malignant_ref_clusters -恶性簇索引
    
    返回:
    best_candidate_results - 最佳k值下的候选域聚类结果
    best_k - 最佳k值
    """
    print(f"  候选细胞数量为 {len(patient_candidate_normalized)}，开始优化k值...")
    
    # 创建优化结果目录
    k_optimization_dir = os.path.join(patient_save_dir, 'k_optimization')
    os.makedirs(k_optimization_dir, exist_ok=True)
    
    # 存储每个k值的评分
    k_scores = []
    
    # 从k=2开始，以步长2递增到max_k（或样本数的1/3，取较小值）
    max_k = min(max_k, len(patient_candidate_normalized) // 3)
    k_values = range(2, max_k + 1, 2)
    # k_values = range(5, 6, 1) # 为了测试

    # 定义恶性和良性参考簇
    # malignant_ref_clusters = [0, 5]  # 恶性簇索引 (R0和R5)
    benign_ref_clusters = [i for i in range(reference_results['n_clusters']) if i not in malignant_ref_clusters]
    
    # 计算每个候选细胞与所有参考簇中心的余弦相似度
    reference_centers = reference_results['cluster_centers']
    cell_similarity_matrix = cosine_similarity(patient_candidate_normalized, reference_centers)
    
    # 对每个细胞，找出最近的参考簇
    cell_nearest_ref_cluster = np.argmin(1 - cell_similarity_matrix, axis=1)
    
    for k in k_values:
        print(f"  尝试k值: {k}")
        
        try:
            # 为当前k值创建保存目录
            k_save_dir = os.path.join(k_optimization_dir, f'k{k}')
            os.makedirs(k_save_dir, exist_ok=True)

            # 对candidate域进行K-Means聚类
            patient_candidate_kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300)
            patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)
            
            # 检查是否有空簇
            cluster_counts = np.bincount(patient_candidate_labels, minlength=k)
            if np.any(cluster_counts == 0):
                print(f"  警告: k={k} 时出现空簇，跳过此k值")
                continue
            
            # 计算candidate域每个样本到其聚类中心的距离
            patient_candidate_distances = np.linalg.norm(
                patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels], 
                axis=1
            )
            
            # 构建candidate结果字典
            current_candidate_results = {
                'features': patient_candidate_normalized,
                'labels': patient_candidate_labels,
                'distances': patient_candidate_distances,
                'source_labels': patient_candidate_source_labels,
                'image_paths': patient_candidate_image_paths,
                'kmeans': patient_candidate_kmeans,
                'cluster_centers': patient_candidate_kmeans.cluster_centers_,
                'n_clusters': k
            }
            
            # 计算相似度矩阵
            candidate_centers = current_candidate_results['cluster_centers']
            similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
            
            # 计算距离矩阵 (1 - 相似度)
            distance_matrix = 1 - similarity_matrix
            
            # 对每个候选簇，找出最近的参考簇
            cluster_nearest_ref = np.argmin(distance_matrix, axis=1)
            
            # 计算每个候选簇的分离度和一致性评分
            cluster_scores = []
            total_cells = len(patient_candidate_normalized)
            
            for cand_idx in range(k):
                # 获取该簇的样本数量
                cluster_mask = (patient_candidate_labels == cand_idx)
                cluster_size = np.sum(cluster_mask)
                
                # a. 计算与最近的恶性参考簇的距离
                malignant_distances = [distance_matrix[cand_idx, ref_idx] for ref_idx in malignant_ref_clusters]
                Dm = min(malignant_distances)  # 最近的恶性簇距离
                
                # b. 计算与最近的良性参考簇的距离
                benign_distances = [distance_matrix[cand_idx, ref_idx] for ref_idx in benign_ref_clusters]
                Db = min(benign_distances) if benign_distances else 1.0  # 最近的良性簇距离
                
                # c. 计算归一化的距离分离度量
                epsilon = 1e-10  # 避免除零
                Ssep = abs((Dm - Db) / (Dm + Db + epsilon))
                
                # d. 分析单细胞最近reference簇与所在candidate簇中心最近reference簇是否匹配
                cluster_cells_mask = (patient_candidate_labels == cand_idx)
                cluster_cells_nearest_ref = cell_nearest_ref_cluster[cluster_cells_mask]
                
                # 该簇中心最近的参考簇
                cluster_nearest_ref_cluster = cluster_nearest_ref[cand_idx]
                
                # 计算该簇中与簇中心最近参考簇相匹配的细胞数量
                matching_cells_count = np.sum(cluster_cells_nearest_ref == cluster_nearest_ref_cluster)
                
                # 计算匹配占比（相对于所有候选细胞）
                P_C = matching_cells_count / total_cells
                
                # 存储该簇的分离度、大小和匹配占比
                cluster_scores.append((Ssep, cluster_size, P_C))
            
            # e. 使用每簇的P(C)为权重计算该K值下总的评分
            total_P_C = sum(P_C for _, _, P_C in cluster_scores)
            if total_P_C > 0:
                weighted_score = sum(Ssep * P_C for Ssep, _, P_C in cluster_scores) / total_P_C
            else:
                weighted_score = 0
            
            print(f"  k={k} 的加权评分: {weighted_score:.4f}")
            k_scores.append((k, weighted_score))
            
            # 保存当前k值的结果，以便后续使用
            np.save(os.path.join(k_save_dir, 'candidate_cluster_centers.npy'), candidate_centers)
            np.save(os.path.join(k_save_dir, 'candidate_labels.npy'), patient_candidate_labels)
            np.save(os.path.join(k_save_dir, 'cluster_scores.npy'), np.array(cluster_scores))
                
        except Exception as e:
            print(f"  尝试k={k}时出错: {str(e)}，跳过此k值")
            continue
    
    # 检查是否找到了有效的k值
    if not k_scores:
        print("  警告: 所有k值尝试都失败，将使用默认k=2")
        # 使用k=2作为最后的备选方案
        try:
            k = 2
            k_save_dir = os.path.join(k_optimization_dir, f'k{k}')
            os.makedirs(k_save_dir, exist_ok=True)
            
            patient_candidate_kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300)
            patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)
            
            patient_candidate_distances = np.linalg.norm(
                patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels], 
                axis=1
            )
            
            best_candidate_results = {
                'features': patient_candidate_normalized,
                'labels': patient_candidate_labels,
                'distances': patient_candidate_distances,
                'source_labels': patient_candidate_source_labels,
                'image_paths': patient_candidate_image_paths,
                'kmeans': patient_candidate_kmeans,
                'cluster_centers': patient_candidate_kmeans.cluster_centers_,
                'n_clusters': k
            }
            
            best_k = k
            print(f"  使用默认k={k}完成聚类")
            
        except Exception as e:
            print(f"  使用默认k=2也失败: {str(e)}，跳过此病理片子")
            return None, None  # 跳过当前病理片子的处理
    else:
        # 选择评分最高的k值
        k_scores.sort(key=lambda x: x[1], reverse=True)  # 按评分降序排序
        best_k, best_score = k_scores[0]
        
        # 保存k值优化结果
        k_df = pd.DataFrame(k_scores, columns=['k', 'weighted_score'])
        k_df.to_csv(os.path.join(k_optimization_dir, 'k_optimization_results.csv'), index=False)
        
        # 可视化k值优化结果
        k_vals = [x[0] for x in k_scores]
        scores = [x[1] for x in k_scores]
        # 找到最佳k (假设你已经计算出 best_k)
        
        # 调用可视化模块
        plot_k_optimization_curve(k_vals, scores, best_k, patient_id, k_optimization_dir)
        
        # 重新运行最佳k的聚类
        patient_candidate_kmeans = KMeans(n_clusters=best_k, random_state=42, max_iter=300)
        patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)
        
        patient_candidate_distances = np.linalg.norm(
            patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels], 
            axis=1
        )
        
        best_candidate_results = {
            'features': patient_candidate_normalized,
            'labels': patient_candidate_labels,
            'distances': patient_candidate_distances,
            'source_labels': patient_candidate_source_labels,
            'image_paths': patient_candidate_image_paths,
            'kmeans': patient_candidate_kmeans,
            'cluster_centers': patient_candidate_kmeans.cluster_centers_,
            'n_clusters': best_k
        }
        
        print(f"  优化完成，最佳k值: {best_k}，最佳加权评分: {best_score:.4f}")
    
    return best_candidate_results, best_k
