"""ShiLab research software utilities."""

__version__ = "0.3.0"
__author__ = "ShiLab"

from shilab_cancer_classifier.data.dataset_splitter import split_dataset, clear_directory
from shilab_cancer_classifier.data.cross_validation import prepare_cross_validation
from shilab_cancer_classifier.training.trainer import train_single_model_fold, train_all_models, set_seed
from shilab_cancer_classifier.models.model_definitions import create_model, SUPPORTED_MODELS
from shilab_cancer_classifier.evaluation.evaluator import evaluate_single_model, evaluate_all_models
from shilab_cancer_classifier.config import CANCER_CLASSES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from shilab_cancer_classifier.infer import predict_matched_malignant_cells

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
    'CANCER_CLASSES',
    'NUM_CLASSES',
    'CLASS_TO_IDX',
    'IDX_TO_CLASS',
    'predict_matched_malignant_cells',
]
