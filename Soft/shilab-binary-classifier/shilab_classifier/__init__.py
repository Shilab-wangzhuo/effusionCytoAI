"""
ShiLab Binary Classifier
========================

ShiLab课题组的二分类模型工具包

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

from shilab_classifier.data.dataset_splitter import split_dataset, clear_directory
from shilab_classifier.data.cross_validation import prepare_cross_validation
from shilab_classifier.training.trainer import train_single_model_fold, train_all_models, set_seed
from shilab_classifier.models.model_definitions import create_model, SUPPORTED_MODELS
from shilab_classifier.evaluation.evaluator import evaluate_single_model, evaluate_all_models

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
    'evaluate_all_models'
]
