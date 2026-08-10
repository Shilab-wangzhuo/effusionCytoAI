#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 pipeline.py 中模型提取部分的代码
针对19个模型进行逐一测试
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse

# import sys
# import os

# # 添加项目根目录到 Python 路径
# sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# 从 pipeline.py 导入所需模块
from src.models.models import get_model_and_transform
from src.data.data_utils import collect_reference_data, collect_image_paths, CellImageDataset,  extract_patient_id
from src.models.feature_extractor import extract_features_with_cell_size
from src.clustering.K_optimizer import optimize_k_for_candidate_clustering
from src.clustering.calculate_consensus_score import calculate_consensus_score
from src.clustering.matching_rule_application import identify_matching_clusters, save_matched_cells, check_matching_rules
from src.visualization.visualization import (
    count_non_background_pixels,
    visualize_all_clusters_without_images, 
    improved_visualize_clusters_with_images, 
    visualize_matching_clusters,
    analyze_cluster_composition,
    plot_similarity_heatmap,
    visualize_filtered_clusters
)

def extract_mean_std_from_log(log_file_path):
    """
    从日志文件中提取均值和标准差
    根据固定行号（第6行和第7行）提取
    """
    mean = None
    std = None
    
    if not os.path.exists(log_file_path):
        print(f"警告: 日志文件不存在: {log_file_path}")
        return mean, std
    
    # 尝试多种编码方式
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    lines = None
    
    for encoding in encodings:
        try:
            with open(log_file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"成功使用 {encoding} 编码读取日志文件")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue
    
    if lines is None:
        print(f"无法使用常见编码读取日志文件: {log_file_path}")
        return mean, std
    
    # 根据固定行号提取（第6行和第7行，索引为5和6）
    # 假设格式是：
    # 第5行：均值描述
    # 第6行：均值值 [0.5884, 0.5755, 0.7305]
    # 第6行：标准差描述  
    # 第7行：标准差值 [0.2668, 0.2546, 0.1377]
    
    try:
        if len(lines) >= 8:  # 确保至少有8行
            # 从第6行和第7行（索引5和6）直接提取
            mean_line = lines[5].strip() if len(lines) > 5 else ""
            std_line = lines[6].strip() if len(lines) > 6 else ""
            
            # 解析均值
            if mean_line:
                start_idx = mean_line.find('[')
                end_idx = mean_line.find(']')
                if start_idx != -1 and end_idx != -1:
                    values_str = mean_line[start_idx+1:end_idx]
                    mean_values = [float(x.strip()) for x in values_str.split(',')]
                    mean = mean_values
            
            # 解析标准差
            if std_line:
                start_idx = std_line.find('[')
                end_idx = std_line.find(']')
                if start_idx != -1 and end_idx != -1:
                    values_str = std_line[start_idx+1:end_idx]
                    std_values = [float(x.strip()) for x in values_str.split(',')]
                    std = std_values
                    
    except Exception as e:
        print(f"解析日志文件时出错: {e}")
        return mean, std
    
    return mean, std



def test_single_model_extraction(
    model_type,
    model_path,
    malignant_cells_dir,
    benign_cells_dir,
    batch_size=8,
    num_workers=0,
    cell_size_weight=1.0,
    feature_layer='penultimate',
):
    """
    测试单个模型的特征提取功能
    """
    print(f"开始测试模型: {model_type}")
    print(f"模型权重路径: {model_path}")
    
    # 尝试从对应的日志文件中读取 mean 和 std
    log_file_path = model_path.replace('.pth', '.log')
    log_mean, log_std = extract_mean_std_from_log(log_file_path)
    
    if log_mean is not None and log_std is not None:
        mean = log_mean
        std = log_std
        print(f"从日志文件 {os.path.basename(log_file_path)} 中读取到均值: {mean}, 标准差: {std}")
    else:
        print(f"未能从日志文件 {os.path.basename(log_file_path)} 中读取到均值和标准差，使用默认值或传入的值")
    
    try:
        # 1. 初始化环境与模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model, transform, input_size = get_model_and_transform(
            model_type, 
            device, 
            model_path,
            mean=mean,
            std=std
        )
        print(f"模型加载成功! 输入尺寸: {input_size}")
        
        # 2. 处理参考域 (Reference Domain)
        print("======= 处理参考细胞 =======")
        ref_paths, ref_sources = collect_reference_data(malignant_cells_dir, benign_cells_dir)
        print(f"参考细胞总数: {len(ref_paths)}")
        
        if len(ref_paths) == 0:
            print("警告: 没有找到参考细胞，跳过此模型测试")
            return False
        
        # 计算大小
        ref_sizes = np.log1p([count_non_background_pixels(p) for p in tqdm(ref_paths[:10], desc="计算参考细胞大小(前10个)")])  # 限制数量加快测试
        
        # 提取特征
        ref_loader = DataLoader(
            CellImageDataset(ref_paths[:10], transform=transform, source_labels=ref_sources[:10]),  # 限制数量加快测试
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        ref_feats, ref_sources, ref_paths = extract_features_with_cell_size(
            model, ref_loader, device, ref_sizes, feature_layer, cell_size_weight
        )
        print(f"参考特征提取完成，特征形状: {ref_feats.shape}")        
        print(f"测试完成: {model_type}")
        return True
        
    except Exception as e:
        print(f"模型 {model_type} 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_models(
    model_dir,
    malignant_cells_dir,
    benign_cells_dir,
    batch_size=8,
    pca_dim=32
):
    """
    测试多个模型
    """
    print("开始测试模型...")
    
    # 支持的模型列表（根据 SUPPORTED_MODELS）
    supported_models = [
        'AlexNet', 'ResNeXt', 'ConvNextLarge', 'GoogLeNet',
        'ResNet50', 'ResNet101', 'DenseNet121', 'DenseNet161',
        'VGG16', 'InceptionV3', 'EfficientNetB0', 'EfficientNetV2Large',
        'MobileNetV2', 'MobileNetV3', 'RegNet', 'ShuffleNetV2',
        'SqueezeNet', 'ViT_B16', 'ViT_L16'
    ]
    
    print(f"\n支持的模型列表 ({len(supported_models)} 个):")
    for i, model in enumerate(supported_models, 1):
        print(f"{i:2d}. {model}")
    
    # 默认测试所有模型
    selected_models = supported_models
    
    print(f"\n将测试以下 {len(selected_models)} 个模型:")
    for model in selected_models:
        print(f"  - {model}")
    
    # 检查模型权重文件
    print("\n正在查找模型权重文件...")
    model_files = {}
    
    for model_name in selected_models:
        # 根据模型名称构建权重文件名模式
        model_file_patterns = [
            # f"{model_name.lower()}_fold_5.pth",
            # f"{model_name.lower()}_fold_1.pth", 
            # f"{model_name.lower()}_fold_2.pth",
            # f"{model_name.lower()}_fold_3.pth",
            f"{model_name.lower()}_fold_4.pth"
        ]
        
        found_file = None
        for pattern in model_file_patterns:
            model_file_path = os.path.join(model_dir, model_name, pattern)
            if os.path.exists(model_file_path):
                found_file = model_file_path
                break
        
        if found_file:
            model_files[model_name] = found_file
            print(f"✓ 找到 {model_name} 的权重文件: {os.path.basename(found_file)}")
        else:
            print(f"✗ 未找到 {model_name} 的权重文件")
    
    # 开始测试
    successful_models = []
    failed_models = []
    
    print(f"\n开始测试 {len(model_files)} 个模型...")
    
    for model_name, model_path in model_files.items():
        print(f"\n{'='*60}")
        print(f"测试 {model_name}")
        print(f"{'='*60}")
        
        success = test_single_model_extraction(
            model_type=model_name,
            model_path=model_path,
            malignant_cells_dir=malignant_cells_dir,
            benign_cells_dir=benign_cells_dir,
            batch_size=batch_size,
            
        )
        
        if success:
            successful_models.append(model_name)
            print(f"✓ {model_name} 测试成功!")
        else:
            failed_models.append(model_name)
            print(f"✗ {model_name} 测试失败!")
    
    # 输出最终结果
    print(f"\n{'='*60}")
    print("测试完成总结:")
    print(f"总共测试: {len(model_files)} 个模型")
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
    
    return successful_models, failed_models


def main():
    print("模型提取功能测试工具")
    print("="*50)
    
    # 配置参数
    model_dir = r"H:\limr\project\urine\result\classification_model\UR_cluster\train_val_models"  # 模型权重目录
    malignant_cells_dir = r"H:\limr\project\urine\rawdata\classification_model\UR_cluster\malignant"  # 恶性细胞目录，需要您填入实际路径
    benign_cells_dir = r"H:\limr\project\urine\rawdata\classification_model\UR_cluster\benign"     # 良性细胞目录，需要您填入实际路径

    
    # 可选参数
    batch_size = 8             # 批次大小
    pca_dim = 32               # PCA维度
    
    try:
        test_multiple_models(
            model_dir=model_dir,
            malignant_cells_dir=malignant_cells_dir,
            benign_cells_dir=benign_cells_dir,
            batch_size=batch_size,
            pca_dim=pca_dim
        )
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()