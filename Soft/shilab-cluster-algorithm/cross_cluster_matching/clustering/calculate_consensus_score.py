# clustering/calculate_consensus_score.py
# 用来存放计算匹配簇的共识得分的函数

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import shutil


# 计算匹配簇的共识得分（双向共识计算）
def calculate_consensus_score(reference_results, candidate_results, matching_pairs, save_dir=None):
    """
    计算匹配簇的共识得分 - 修改版，适用于新的匹配规则和双向共识计算
    
    参数:
    reference_results: 参考域结果字典
    candidate_results: 候选域结果字典
    matching_pairs: 匹配对列表，每个元素为(ref_idx, cand_idx, similarity)
    save_dir: 保存结果的目录
    
    返回:
    consensus_scores: 每个匹配对的共识得分
    """
    # 提取参考域和候选域的特征和标签
    reference_features = reference_results['features']
    reference_labels = reference_results['labels']
    candidate_features = candidate_results['features']
    candidate_labels = candidate_results['labels']
    
    # 提取参考域中的恶性簇索引（R0和R5）
    malignant_ref_clusters = [0, 5]
    
    # 计算候选域中每个样本到参考域中R0和R5簇中心的余弦相似度
    ref_centers = reference_results['cluster_centers']
    
    # 分别获取R0和R5的簇中心
    R0_center = ref_centers[0].reshape(1, -1)  # 确保是2D数组
    R5_center = ref_centers[5].reshape(1, -1)  # 确保是2D数组
    
    # 计算候选域每个样本到R0和R5簇中心的余弦相似度
    cell_to_R0_similarity = cosine_similarity(candidate_features, R0_center)
    cell_to_R5_similarity = cosine_similarity(candidate_features, R5_center)
    
    # 计算参考域中R0和R5的样本到候选域所有簇中心的余弦相似度
    cand_centers = candidate_results['cluster_centers']
    
    # 获取参考域中R0和R5的样本索引
    R0_indices = np.where(reference_labels == 0)[0]
    R5_indices = np.where(reference_labels == 5)[0]
    
    # 获取R0和R5的样本特征
    R0_features = reference_features[R0_indices]
    R5_features = reference_features[R5_indices]
    
    # 为每个匹配对计算共识得分
    consensus_scores = []
    
    for ref_idx, cand_idx, similarity in matching_pairs:
        # 只关注与R0或R5匹配的候选簇
        if ref_idx in malignant_ref_clusters:
            # 获取该候选簇的所有样本
            cand_cluster_indices = np.where(candidate_labels == cand_idx)[0]
            
            if len(cand_cluster_indices) > 0:
                # 1. 计算候选域样本与R0和R5的相似度
                
                # 计算该簇中与R0最相似的样本数量
                closest_to_R0_count = 0
                # 计算该簇中与R5最相似的样本数量
                closest_to_R5_count = 0
                
                for idx in cand_cluster_indices:
                    # 比较该样本与R0和R5的相似度
                    if cell_to_R0_similarity[idx] > cell_to_R5_similarity[idx]:
                        closest_to_R0_count += 1
                    else:
                        closest_to_R5_count += 1
                
                # 计算候选域共识得分：与R0或R5最接近的样本比例
                candidate_consensus_score = (closest_to_R0_count + closest_to_R5_count) / len(cand_cluster_indices)
                
                # 2. 反向计算：参考域R0和R5样本与当前候选簇的相似度
                
                # 获取当前候选簇的中心
                current_cand_center = cand_centers[cand_idx].reshape(1, -1)
                
                # 计算R0样本到当前候选簇中心的相似度
                R0_to_current_cand_similarity = cosine_similarity(R0_features, current_cand_center)
                
                # 计算R0样本到所有候选簇中心的相似度
                R0_to_all_cand_similarity = cosine_similarity(R0_features, cand_centers)
                
                # 计算R0中最接近当前候选簇的样本数量
                R0closest_to_CurrentCandidate_count = 0
                for i in range(len(R0_indices)):
                    # 找出该R0样本最接近的候选簇
                    closest_cand = np.argmax(R0_to_all_cand_similarity[i])
                    # 如果最接近的是当前候选簇，计数加1
                    if closest_cand == cand_idx:
                        R0closest_to_CurrentCandidate_count += 1
                
                # 计算R5样本到当前候选簇中心的相似度
                R5_to_current_cand_similarity = cosine_similarity(R5_features, current_cand_center)
                
                # 计算R5样本到所有候选簇中心的相似度
                R5_to_all_cand_similarity = cosine_similarity(R5_features, cand_centers)
                
                # 计算R5中最接近当前候选簇的样本数量
                R5closest_to_CurrentCandidate_count = 0
                for i in range(len(R5_indices)):
                    # 找出该R5样本最接近的候选簇
                    closest_cand = np.argmax(R5_to_all_cand_similarity[i])
                    # 如果最接近的是当前候选簇，计数加1
                    if closest_cand == cand_idx:
                        R5closest_to_CurrentCandidate_count += 1
                
                # 计算参考域共识得分
                if len(R0_indices) + len(R5_indices) > 0:
                    reference_consensus_score = (R0closest_to_CurrentCandidate_count + R5closest_to_CurrentCandidate_count) / (len(R0_indices) + len(R5_indices))
                else:
                    reference_consensus_score = 0
                
                # 计算总共识得分：候选域共识得分和参考域共识得分的平均值
                consensus_score = (candidate_consensus_score + reference_consensus_score) / 2
                
                # 添加到结果列表
                consensus_scores.append({
                    'Reference_Cluster': ref_idx,
                    'Candidate_Cluster': cand_idx,
                    'Cluster_Similarity': similarity,
                    'Cluster_Size': len(cand_cluster_indices),
                    'Cells_Closest_To_R0': closest_to_R0_count,
                    'Cells_Closest_To_R5': closest_to_R5_count,
                    'Cells_Closest_To_Malignant': closest_to_R0_count + closest_to_R5_count,
                    'Candidate_Consensus_Score': candidate_consensus_score,
                    'R0Cells_Closest_To_CurrentCandidate': R0closest_to_CurrentCandidate_count,
                    'R5Cells_Closest_To_CurrentCandidate': R5closest_to_CurrentCandidate_count,
                    'R0R5Cells_Closest_To_CurrentCandidate': R0closest_to_CurrentCandidate_count + R5closest_to_CurrentCandidate_count,
                    'Reference_Consensus_Score': reference_consensus_score,
                    'Consensus_Score': consensus_score,
                })
    
    # 如果有保存目录，保存结果
    if save_dir and consensus_scores:
        consensus_df = pd.DataFrame(consensus_scores)
        consensus_df.to_csv(os.path.join(save_dir, 'cluster_matching.csv'), index=False)
        print(f"共识得分已保存到 {os.path.join(save_dir, 'cluster_matching.csv')}")
    
    return consensus_scores
