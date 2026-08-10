"""
ShiLab Cancer Classifier
========================

ShiLab课题组的癌症分类模型工具包

主要功能:
- 数据集分割（训练集、验证集、测试集）
- K折交叉验证数据准备
- 19个SOTA模型训练
- 模型评估与性能分析
- 更多功能开发中...

作者: ShiLab
版本: 0.3.0
"""

__version__ = "0.3.0"
__author__ = "ShiLab"

from shilab_cancer_classifier.data.dataset_splitter import split_dataset, clear_directory
from shilab_cancer_classifier.data.cross_validation import prepare_cross_validation
from shilab_cancer_classifier.training.trainer import train_single_model_fold, train_all_models, set_seed
from shilab_cancer_classifier.models.model_definitions import create_model, SUPPORTED_MODELS
from shilab_cancer_classifier.evaluation.evaluator import evaluate_single_model, evaluate_all_models
from shilab_cancer_classifier.config import CANCER_CLASSES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from shilab_cancer_classifier.infer import predict_matched_malignant_cells
from shilab_cancer_classifier.clustering.model import get_model_and_transform
from shilab_cancer_classifier.clustering.visualization import (
    visualize_all_clusters_without_images,
    visualize_all_clusters_without_images_1,
    improved_visualize_clusters_with_images,
    visualize_matching_clusters,
    analyze_cluster_composition,
    plot_similarity_heatmap,
    visualize_filtered_clusters,
    build_category_config
)

__all__ = [
    'split_dataset',
    'clear_directory',
    'prepare_cross_validation',
    'train_single_model_fold',
    'train_all_models',
    'set_seed',
    'create_model',
    'SUPPORTED_MODELS',
    'evaluate_single_model',
    'evaluate_all_models',
    'get_model_and_transform',
    'visualize_all_clusters_without_images',
    'visualize_all_clusters_without_images_1',
    'improved_visualize_clusters_with_images',
    'visualize_all_clusters_without_images',
    'visualize_matching_clusters',
    'analyze_cluster_composition',
    'plot_similarity_heatmap',
    'visualize_filtered_clusters',
    'build_category_config',
    'CANCER_CLASSES',
    'NUM_CLASSES',
    'CLASS_TO_IDX',
    'IDX_TO_CLASS',
    'predict_matched_malignant_cells',
]
