# candidate_only_clustering.py
# 仅对候选域细胞进行聚类分析，使用自定义K值

import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import argparse

from cross_cluster_matching.models.models import get_model_and_transform
from cross_cluster_matching.data.data_utils import collect_image_paths, CellImageDataset, extract_patient_id
from cross_cluster_matching.models.feature_extractor import extract_features_with_cell_size
from cross_cluster_matching.visualization.visualization import (
    count_non_background_pixels,
    visualize_all_clusters_without_images,
    improved_visualize_clusters_with_images,
    analyze_cluster_composition
)

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_patient_folders(base_dir, cell_type="single_cell"):
    """
    查找所有病人文件夹，返回包含指定cell_type的完整路径列表
    
    参数:
        base_dir: 基础目录
        cell_type: 细胞类型目录名称，可以是'single_cell'或'cluster'
    """
    patient_folders = []
    
    # 遍历基础目录下的所有病人文件夹
    for patient_folder in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_folder)
        
        # 确保是目录而不是文件
        if not os.path.isdir(patient_path):
            continue
            
        # 查找confidence_0.4/cell_type路径
        cell_path = os.path.join(patient_path, "confidence_0.4", cell_type)
        
        # 检查路径是否存在且包含图片
        if os.path.isdir(cell_path) and any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')) 
                                           for f in os.listdir(cell_path)):
            patient_folders.append({
                'name': patient_folder,
                'path': cell_path
            })
    
    return patient_folders

def run_candidate_clustering(
    candidate_folder_path,
    model_path,
    save_dir,
    k_value=10,  # 自定义的K值，不使用优化方法
    feature_layer='penultimate',
    batch_size=16,
    num_workers=0,
    cell_size_weight=1.0,
    model_type='ResNeXt',
    pca_dim=32,
    mean=None,
    std=None,
    save_cluster_images=True  # 是否保存聚类后的图像
):
    """
    仅对候选域细胞进行聚类分析，使用自定义K值
    
    参数:
    candidate_folder_path: 包含候选细胞图像的文件夹路径
    model_path: 预训练模型文件路径
    save_dir: 结果保存目录
    k_value: 聚类的K值，由用户指定
    feature_layer: 用于特征提取的模型层，默认为'penultimate'
    batch_size: 数据加载批次大小，默认为16
    num_workers: 数据加载进程数，默认为0
    cell_size_weight: 细胞大小权重因子，默认为1.0
    model_type: 使用的模型类型，默认为'ResNeXt'
    pca_dim: PCA降维后的维度，默认为32
    mean: 图像归一化的均值，默认为None
    std: 图像归一化的标准差，默认为None
    save_cluster_images: 是否将聚类后的图像保存到对应簇的文件夹，默认为True
    """
    set_seed()
    print(f"开始对候选域细胞进行聚类分析，K值设置为: {k_value}")
    print(f"使用均值: {mean}, 标准差: {std}")
    
    # 1. 初始化环境与模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model, transform, _ = get_model_and_transform(model_type, device, model_path, mean=mean, std=std)
    
    # 2. 处理候选域 (Candidate Domain)
    print("======= 处理候选域细胞 =======")
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集候选域细胞图像路径
    cand_paths = collect_image_paths(candidate_folder_path)
    if not cand_paths:
        print(f"错误: 在 {candidate_folder_path} 中未找到图像文件")
        return
    
    print(f"找到 {len(cand_paths)} 个候选域细胞图像")
    cand_sources = [0] * len(cand_paths)  # 所有候选域细胞的源标签都是0
    
    # 计算细胞大小
    print("计算细胞大小...")
    cand_sizes = np.log1p([count_non_background_pixels(p) for p in tqdm(cand_paths, desc="计算细胞大小")])
    
    # 提取特征
    print("提取特征...")
    cand_loader = DataLoader(
        CellImageDataset(cand_paths, transform=transform, source_labels=cand_sources),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    cand_feats, _, cand_paths = extract_features_with_cell_size(
        model, cand_loader, device, cand_sizes, feature_layer, cell_size_weight
    )
    
    # PCA降维
    print(f"执行PCA降维到 {pca_dim} 维...")
    pca = PCA(n_components=min(pca_dim, cand_feats.shape[0], cand_feats.shape[1]))
    cand_norm = normalize(pca.fit_transform(cand_feats))
    
    # K-Means聚类
    print(f"执行K-Means聚类，K = {k_value}...")
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10).fit(cand_norm)
    
    # 保存聚类结果
    cand_results = {
        'features': cand_norm,
        'labels': kmeans.labels_,
        'source_labels': cand_sources,
        'image_paths': cand_paths,
        'cluster_centers': kmeans.cluster_centers_,
        'n_clusters': k_value,
        'pca': pca
    }
    
    # 保存聚类标签到CSV
    print("保存聚类结果到CSV...")
    csv_data = {
        'image_path': cand_paths,
        'cluster_label': kmeans.labels_
    }
    df = pd.DataFrame(csv_data)
    csv_save_path = os.path.join(save_dir, 'cluster_labels.csv')
    df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
    
    # 可视化聚类结果
    print("生成聚类可视化...")
    
    # 1. 不带图像的聚类可视化（UMAP降维）
    umap_plot = visualize_all_clusters_without_images(
        cand_results['features'],
        cand_results['labels'],
        cand_results['source_labels'],
        f"Candidate Clustering (K={k_value})",
        save_dir
    )
    
    # 2. 带图像的聚类可视化
    improved_visualize_clusters_with_images(
        cand_results['features'],
        cand_results['labels'],
        cand_results['source_labels'],
        cand_results['image_paths'],
        f"Candidate Clustering (K={k_value})",
        save_dir,
        cand_sizes
    )
    
    # 3. 分析聚类组成
    analyze_cluster_composition(
        cand_results['labels'],
        cand_results['source_labels'],
        cand_results['image_paths'],
        f"Candidate Clustering (K={k_value})",
        save_dir
    )
    
    # 4. 统计每个簇的细胞数量并绘图
    cluster_counts = np.bincount(cand_results['labels'], minlength=k_value)
    plt.figure(figsize=(12, 6))
    plt.bar(range(k_value), cluster_counts)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Cells')
    plt.title(f'Cell Count per Cluster (K={k_value})')
    plt.xticks(range(k_value))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'cluster_cell_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可选：将聚类后的图像复制到对应簇的文件夹中
    if save_cluster_images:
        print("将聚类后的图像保存到对应簇的文件夹...")
        clusters_images_dir = os.path.join(save_dir, "cluster_images")
        os.makedirs(clusters_images_dir, exist_ok=True)
        
        # 为每个簇创建目录
        for cluster_id in range(k_value):
            cluster_dir = os.path.join(clusters_images_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)
        
        # 复制图像到对应目录
        for img_path, label in tqdm(zip(cand_results['image_paths'], cand_results['labels']),
                                   total=len(cand_results['image_paths']),
                                   desc="复制图像到簇文件夹"):
            # 获取目标目录
            cluster_num = label
            target_dir = os.path.join(clusters_images_dir, f"cluster_{cluster_num}")
            
            # 获取文件名并复制
            file_name = Path(img_path).name
            target_path = os.path.join(target_dir, file_name)
            
            # 复制文件
            shutil.copy2(img_path, target_path)
    
    print(f"候选域聚类分析完成，结果已保存到: {save_dir}")
    return cand_results

def process_patient_folders(
    base_folder_path,
    model_path,
    base_save_dir,
    cell_type="single_cell",
    mean=None,
    std=None,
    k_value=10,
    **kwargs
):
    """
    处理病人文件夹结构中的细胞图像
    
    参数:
    base_folder_path: 包含多个病人文件夹的基础路径
    model_path: 预训练模型文件路径
    base_save_dir: 结果保存的基础目录
    cell_type: 细胞类型，可以是'single_cell'或'cluster'
    mean: 图像归一化的均值
    std: 图像归一化的标准差
    k_value: 聚类的K值，由用户指定
    **kwargs: 传递给run_candidate_clustering的其他参数
    """
    set_seed()
    
    # 确保保存目录存在
    os.makedirs(base_save_dir, exist_ok=True)
    
    print(f"均值: {mean}, 标准差: {std}")
    print(f"开始处理病人文件夹: {base_folder_path}")
    print(f"查找细胞类型: {cell_type}")
    
    # 查找所有病人文件夹
    patient_folders = find_patient_folders(base_folder_path, cell_type)
    
    if not patient_folders:
        print(f"在 {base_folder_path} 中未找到任何包含 {cell_type} 的病人文件夹")
        return
    
    print(f"找到 {len(patient_folders)} 个病人文件夹")
    
    # 处理每个病人文件夹
    results = []
    for idx, patient in enumerate(patient_folders, 1):
        patient_name = patient['name']
        patient_path = patient['path']
        
        # 创建病人特定的输出目录：base_save_dir/patient_name/cell_type
        patient_save_dir = os.path.join(base_save_dir, patient_name)
        os.makedirs(patient_save_dir, exist_ok=True)
        
        cell_type_save_dir = os.path.join(patient_save_dir, cell_type)
        
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(patient_folders)}] 处理病人: {patient_name} ({cell_type})")
        print(f"  - 输入路径: {patient_path}")
        print(f"  - 输出路径: {cell_type_save_dir}")
        print(f"{'='*60}")
        
        result = run_candidate_clustering(
            candidate_folder_path=patient_path,
            model_path=model_path,
            save_dir=cell_type_save_dir,
            mean=mean,
            std=std,
            k_value=k_value,
            **kwargs
        )
        
        if result:
            results.append({
                'patient': patient_name,
                'result': result
            })
    
    print(f"\n{'='*60}")
    print(f"所有病人处理完成，共处理 {len(results)}/{len(patient_folders)} 个病人")
    print(f"结果保存在: {base_save_dir}")
    
    return results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='候选域细胞聚类分析')
    
    # 必需参数
    parser.add_argument('--candidate_folder', type=str, required=True, 
                        help='包含候选细胞图像的文件夹路径')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='预训练模型文件路径')
    parser.add_argument('--save_dir', type=str, required=True, 
                        help='结果保存的基础目录')
    
    # 可选参数
    parser.add_argument('--k_value', type=int, default=10, 
                        help='聚类的K值，默认为10')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='数据加载批次大小，默认为16')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='数据加载进程数，默认为0')
    parser.add_argument('--cell_size_weight', type=float, default=1.0, 
                        help='细胞大小权重因子，默认为1.0')
    parser.add_argument('--model_type', type=str, default='DenseNet161', 
                        help='使用的模型类型，默认为DenseNet161')
    parser.add_argument('--pca_dim', type=int, default=32, 
                        help='PCA降维后的维度，默认为32')
    parser.add_argument('--feature_layer', type=str, default='penultimate', 
                        help='用于特征提取的模型层，默认为penultimate')
    parser.add_argument('--mean', type=float, nargs=3, default=None, 
                        help='图像归一化的均值，格式为三个浮点数，例如: 0.485 0.456 0.406')
    parser.add_argument('--std', type=float, nargs=3, default=None, 
                        help='图像归一化的标准差，格式为三个浮点数，例如: 0.229 0.224 0.225')
    parser.add_argument('--save_cluster_images', action='store_true', 
                        help='是否将聚类后的图像保存到对应簇的文件夹')
    
    # 细胞类型参数
    parser.add_argument('--cell_type', type=str, default="single_cell", choices=["single_cell", "cluster"],
                        help='细胞类型 (默认: single_cell, 可选: cluster)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 如果提供了均值和标准差，将其转换为列表
    mean = args.mean if args.mean else None
    std = args.std if args.std else None
    
    print("="*80)
    print(f"开始运行候选域聚类分析")
    print(f"候选文件夹: {args.candidate_folder}")
    print(f"模型路径: {args.model_path}")
    print(f"保存目录: {args.save_dir}")
    print(f"聚类K值: {args.k_value}")
    print(f"均值: {mean}")
    print(f"标准差: {std}")
    print(f"模型类型: {args.model_type}")
    print(f"细胞类型: {args.cell_type}")
    
    # 处理病人文件夹结构
    process_patient_folders(
        base_folder_path=args.candidate_folder,
        model_path=args.model_path,
        base_save_dir=args.save_dir,
        cell_type=args.cell_type,
        mean=mean,
        std=std,
        k_value=args.k_value,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cell_size_weight=args.cell_size_weight,
        model_type=args.model_type,
        pca_dim=args.pca_dim,
        feature_layer=args.feature_layer,
        save_cluster_images=args.save_cluster_images
    )