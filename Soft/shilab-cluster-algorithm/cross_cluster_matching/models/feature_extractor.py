# models/feature_extractor.py 
# 用来存放提取模型特征向量的函数

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
from tqdm import tqdm


# 从模型中提取特征并添加细胞大小信息
def extract_features_with_cell_size(model, dataloader, device, cell_sizes, feature_layer='penultimate', cell_size_weight=1.0):
    """
    从模型中提取特征，并将细胞大小信息添加到特征向量中
    
    参数:
    model: 特征提取模型
    dataloader: 数据加载器
    device: 设备(CPU/GPU)
    cell_sizes: 细胞大小列表
    feature_layer: 要提取特征的层
    cell_size_weight: 细胞大小特征的权重因子
    
    返回:
    features: 包含细胞大小信息的特征向量
    source_labels: 源标签
    image_paths: 图像路径列表（字符串列表）
    """
    model.eval()
    features_list = []
    source_labels_list = []
    indices_list = []
    
    with torch.no_grad():
        for batch_images, batch_labels, batch_indices in tqdm(dataloader, desc="提取特征"):
            batch_images = batch_images.to(device)
            batch_features = model(batch_images, layer=feature_layer)
            
            # 如果特征是4D张量 (batch_size, channels, height, width)，将其平均池化为2D张量
            if len(batch_features.shape) == 4:
                batch_features = torch.mean(batch_features, dim=(2, 3))
            
            # 将特征移到CPU并转换为NumPy数组
            batch_features = batch_features.cpu().numpy()
            
            features_list.append(batch_features)
            source_labels_list.append(batch_labels.numpy())
            indices_list.append(batch_indices.numpy())
    
    # 连接所有批次的特征和标签
    features = np.vstack(features_list)
    source_labels = np.concatenate(source_labels_list)
    indices = np.concatenate(indices_list)
    
    # 获取原始图像路径列表（确保是字符串列表）
    image_paths = [dataloader.dataset.image_paths[idx] for idx in indices]
    
    # 对应于indices的细胞大小
    batch_cell_sizes = np.array([cell_sizes[idx] for idx in indices])
    
    # 对细胞大小进行对数变换
    log_cell_sizes = np.log1p(batch_cell_sizes)
    
    # 归一化对数变换后的细胞大小
    if np.max(log_cell_sizes) > np.min(log_cell_sizes):  # 避免除以零
        normalized_log_cell_sizes = (log_cell_sizes - np.min(log_cell_sizes)) / (np.max(log_cell_sizes) - np.min(log_cell_sizes))
    else:
        normalized_log_cell_sizes = np.zeros_like(log_cell_sizes)
    
    # 将细胞大小添加到特征向量中
    # 使用权重因子来调整细胞大小特征的重要性
    cell_size_features = cell_size_weight * normalized_log_cell_sizes.reshape(-1, 1)
    features_with_size = np.hstack((features, cell_size_features))
    
    return features_with_size, source_labels, image_paths