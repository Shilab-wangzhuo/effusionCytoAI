"""
评估指标计算模块
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, matthews_corrcoef, roc_curve, auc
)


"""
辅助函数模块

"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    计算各种性能指标
    
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    y_prob: 预测概率 (用于计算AUC)
    
    返回:
    包含各种指标的字典
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 确保混淆矩阵是2x2的
    if cm.shape != (2, 2):
        print(f"警告: 混淆矩阵形状不是(2,2), 而是{cm.shape}")
        if cm.shape[0] == 2 and cm.shape[1] == 1:
            # 可能缺少某个类别的预测，扩展矩阵
            cm = np.column_stack((cm, np.zeros(2)))
        elif cm.shape[0] == 1 and cm.shape[1] == 2:
            # 可能缺少某个类别的真实标签，扩展矩阵
            cm = np.row_stack((cm, np.zeros(2)))
        else:
            # 其他情况，返回零值
            return {
                'accuracy': 0,
                'balanced_accuracy': 0,
                'sensitivity': 0,
                'specificity': 0,
                'precision_benign': 0,
                'recall_benign': 0,
                'f1_benign': 0,
                'precision_malignant': 0,
                'recall_malignant': 0,
                'f1_malignant': 0,
                'mcc': 0,
                'auc': 0 if y_prob is not None else None,
                'benign_correct': 0,
                'benign_total': 0,
                'benign_acc': 0,
                'malignant_correct': 0,
                'malignant_total': 0,
                'malignant_acc': 0,
                'confusion_matrix': np.zeros((2, 2))
            }
    
    # 提取混淆矩阵的各个元素
    # 假设: 0=良性(benign), 1=恶性(malignant)
    tn, fp = cm[0, 0], cm[0, 1]  # 真阴性，假阳性
    fn, tp = cm[1, 0], cm[1, 1]  # 假阴性，真阳性
    
    # 计算基本指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # 计算每个类别的准确率
    benign_correct = tn
    benign_total = tn + fp
    benign_acc = tn / benign_total if benign_total > 0 else 0
    
    malignant_correct = tp
    malignant_total = tp + fn
    malignant_acc = tp / malignant_total if malignant_total > 0 else 0
    
    # 计算灵敏度和特异性
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 灵敏度 = 真阳性率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性 = 真阴性率
    
    # 计算平衡准确率
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # 计算精确度、召回率和F1分数
    precision_benign = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_benign = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_benign = 2 * precision_benign * recall_benign / (precision_benign + recall_benign) if (precision_benign + recall_benign) > 0 else 0
    
    precision_malignant = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_malignant = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_malignant = 2 * precision_malignant * recall_malignant / (precision_malignant + recall_malignant) if (precision_malignant + recall_malignant) > 0 else 0
    
    # 计算MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 计算AUC (如果提供了概率)
    auc_value = None
    if y_prob is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_value = auc(fpr, tpr)
        except Exception as e:
            print(f"计算AUC时出错: {e}")
            auc_value = 0
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision_benign': precision_benign,
        'recall_benign': recall_benign,
        'f1_benign': f1_benign,
        'precision_malignant': precision_malignant,
        'recall_malignant': recall_malignant,
        'f1_malignant': f1_malignant,
        'mcc': mcc,
        'auc': auc_value,
        'benign_correct': int(benign_correct),
        'benign_total': int(benign_total),
        'benign_acc': benign_acc,
        'malignant_correct': int(malignant_correct),
        'malignant_total': int(malignant_total),
        'malignant_acc': malignant_acc,
        'confusion_matrix': cm
    }


def print_metrics(metrics, dataset_name="Dataset"):
    """
    打印评估指标
    
    参数:
    metrics: calculate_metrics返回的指标字典
    dataset_name: 数据集名称
    """
    print(f"\n{dataset_name}:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"平衡准确率: {metrics['balanced_accuracy']:.4f}")
    print(f"灵敏度: {metrics['sensitivity']:.4f}")
    print(f"特异性: {metrics['specificity']:.4f}")
    print(f"良性精确度: {metrics['precision_benign']:.4f}")
    print(f"良性召回率: {metrics['recall_benign']:.4f}")
    print(f"良性F1分数: {metrics['f1_benign']:.4f}")
    print(f"恶性精确度: {metrics['precision_malignant']:.4f}")
    print(f"恶性召回率: {metrics['recall_malignant']:.4f}")
    print(f"恶性F1分数: {metrics['f1_malignant']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    if metrics['auc'] is not None:
        print(f"AUC: {metrics['auc']:.4f}")
    print(f"良性准确率: {metrics['benign_acc']:.4f} ({metrics['benign_correct']}/{metrics['benign_total']})")
    print(f"恶性准确率: {metrics['malignant_acc']:.4f} ({metrics['malignant_correct']}/{metrics['malignant_total']})")
    print(f"混淆矩阵:\n{metrics['confusion_matrix']}")
