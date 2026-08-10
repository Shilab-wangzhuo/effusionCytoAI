"""Metric helpers for multi-class image classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from shilab_cancer_classifier.config import CANCER_CLASSES


def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """Calculate multi-class metrics and per-class statistics."""
    class_names = list(class_names or CANCER_CLASSES)
    labels = list(range(len(class_names)))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length; got "
            f"{y_true.shape[0]} and {y_pred.shape[0]}"
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
    )

    total_samples = int(cm.sum())
    per_class = {}
    for idx, class_name in enumerate(class_names):
        correct = int(cm[idx, idx])
        row_sum = int(cm[idx, :].sum())  # TP + FN, same definition as support.
        col_sum = int(cm[:, idx].sum())  # TP + FP.
        false_positive = col_sum - correct
        true_negative = total_samples - row_sum - col_sum + correct
        specificity_denominator = true_negative + false_positive
        specificity = (
            true_negative / specificity_denominator
            if specificity_denominator > 0 else 0.0
        )

        per_class[class_name] = {
            'precision': float(precision_per_class[idx]),
            'recall': float(recall_per_class[idx]),
            'specificity': float(specificity),
            'f1': float(f1_per_class[idx]),
            'support': int(support_per_class[idx]),
            'correct': correct,
        }

    auc_macro_ovr = None
    auc_weighted_ovr = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        expected_shape = (y_true.shape[0], len(class_names))
        if y_prob.ndim != 2 or y_prob.shape != expected_shape:
            raise ValueError(
                f"y_prob must have shape {expected_shape}; got {y_prob.shape}"
            )

        try:
            auc_macro_ovr = roc_auc_score(
                y_true, y_prob, labels=labels, multi_class='ovr', average='macro'
            )
            auc_weighted_ovr = roc_auc_score(
                y_true, y_prob, labels=labels, multi_class='ovr', average='weighted'
            )
        except ValueError as exc:
            print(f"Could not calculate multi-class AUC: {exc}")
            auc_macro_ovr = None
            auc_weighted_ovr = None

    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'mcc': float(mcc),
        'auc_macro_ovr': auc_macro_ovr,
        'auc_weighted_ovr': auc_weighted_ovr,
        'per_class': per_class,
        'confusion_matrix': cm,
        'class_names': class_names,
    }


def flatten_metrics(metrics, prefix=''):
    """Flatten metrics into a CSV-friendly dictionary."""
    row = {
        f'{prefix}accuracy': metrics['accuracy'],
        f'{prefix}balanced_accuracy': metrics['balanced_accuracy'],
        f'{prefix}precision_macro': metrics['precision_macro'],
        f'{prefix}recall_macro': metrics['recall_macro'],
        f'{prefix}f1_macro': metrics['f1_macro'],
        f'{prefix}precision_weighted': metrics['precision_weighted'],
        f'{prefix}recall_weighted': metrics['recall_weighted'],
        f'{prefix}f1_weighted': metrics['f1_weighted'],
        f'{prefix}mcc': metrics['mcc'],
        f'{prefix}auc_macro_ovr': metrics['auc_macro_ovr'],
        f'{prefix}auc_weighted_ovr': metrics['auc_weighted_ovr'],
    }
    for class_name, values in metrics['per_class'].items():
        safe_name = class_name.lower()
        for metric_name, metric_value in values.items():
            row[f'{prefix}{safe_name}_{metric_name}'] = metric_value
    return row


def print_metrics(metrics, dataset_name="Dataset"):
    """Print a compact multi-class metric summary."""
    print(f"\n{dataset_name}:")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"Weighted F1:       {metrics['f1_weighted']:.4f}")
    print(f"MCC:               {metrics['mcc']:.4f}")
    if metrics['auc_macro_ovr'] is not None:
        print(f"Macro OvR AUC:     {metrics['auc_macro_ovr']:.4f}")
    print("Per-class metrics:")
    for class_name, values in metrics['per_class'].items():
        print(
            f"  {class_name}: recall={values['recall']:.4f}, "
            f"specificity={values['specificity']:.4f}, "
            f"f1={values['f1']:.4f} "
            f"({values['correct']}/{values['support']})"
        )
    print(f"Confusion matrix:\n{metrics['confusion_matrix']}")
