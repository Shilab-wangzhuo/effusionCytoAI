"""
模型评估模块
"""

import os
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from datetime import datetime

from shilab_classifier.models.model_definitions import create_model
from shilab_classifier.training.preprocessing import (
    pad_to_square_transform,
    pad_to_square_299_transform,
    calculate_mean_std
)
from shilab_classifier.evaluation.metrics import calculate_metrics, print_metrics


def predict_and_save_probabilities(model, dataloader, dataset_name, save_dir, device='cuda'):
    """
    对数据集进行预测并保存详细结果
    
    参数:
        model: 训练好的模型
        dataloader: 数据加载器
        dataset_name: 数据集名称 (train/val/test)
        save_dir: 保存目录
        device: 设备
        
    返回:
        dict: 性能指标字典
    """
    model.eval()
    
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []
    
    print(f"为 {dataset_name} 预测概率并保存结果...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"预测 {dataset_name} 批次 {batch_idx}/{len(dataloader)}")
            
            inputs = inputs.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 处理可能的多输出情况
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 计算概率
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # 保存结果
            all_logits.append(outputs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # 保存图片路径
            if hasattr(dataloader.dataset, 'samples'):
                batch_paths = [dataloader.dataset.samples[i][0] for i in range(
                    batch_idx * dataloader.batch_size,
                    min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
                )]
                all_paths.extend(batch_paths)
    
    # 合并所有批次的结果
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 创建DataFrame保存详细结果
    results_df = pd.DataFrame({
        'image_path': all_paths if all_paths else [''] * len(all_labels),
        'filename': [os.path.basename(p) if p else '' for p in all_paths] if all_paths else [''] * len(all_labels),
        'logit_benign': all_logits[:, 0],
        'logit_malignant': all_logits[:, 1],
        'prob_benign': all_probs[:, 0],
        'prob_malignant': all_probs[:, 1],
        'predicted_class': all_preds,
        'true_label': all_labels
    })
    
    # 保存概率结果
    prob_save_path = os.path.join(save_dir, f'{dataset_name}_probabilities.csv')
    results_df.to_csv(prob_save_path, index=False)
    print(f"概率结果已保存到 {prob_save_path}")
    
    # 计算性能指标
    metrics = calculate_metrics(all_labels, all_preds, all_probs[:, 1])
    
    # 保存性能指标
    metrics_df = pd.DataFrame([{
        'dataset': dataset_name,
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'precision_benign': metrics['precision_benign'],
        'recall_benign': metrics['recall_benign'],
        'f1_benign': metrics['f1_benign'],
        'precision_malignant': metrics['precision_malignant'],
        'recall_malignant': metrics['recall_malignant'],
        'f1_malignant': metrics['f1_malignant'],
        'mcc': metrics['mcc'],
        'auc': metrics['auc'],
        'benign_correct': metrics['benign_correct'],
        'benign_total': metrics['benign_total'],
        'benign_acc': metrics['benign_acc'],
        'malignant_correct': metrics['malignant_correct'],
        'malignant_total': metrics['malignant_total'],
        'malignant_acc': metrics['malignant_acc']
    }])
    
    metrics_save_path = os.path.join(save_dir, f'{dataset_name}_metrics.csv')
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"性能指标已保存到 {metrics_save_path}")
    
    # 绘制并保存混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_save_path = os.path.join(save_dir, f'{dataset_name}_confusion_matrix.png')
    plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到 {cm_save_path}")
    
    return metrics


def plot_roc_curve(train_probs, train_labels, val_probs, val_labels, 
                   test_probs, test_labels, save_path):
    """
    绘制ROC曲线
    
    参数:
        train_probs: 训练集预测概率
        train_labels: 训练集真实标签
        val_probs: 验证集预测概率
        val_labels: 验证集真实标签
        test_probs: 测试集预测概率
        test_labels: 测试集真实标签
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 计算ROC曲线
    datasets = [
        ('train', train_probs, train_labels, 'blue'),
        ('val', val_probs, val_labels, 'green'),
        ('test', test_probs, test_labels, 'red')
    ]
    
    for name, probs, labels, color in datasets:
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC = {roc_auc:.4f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存到 {save_path}")


def evaluate_single_model(model_name, cv_data_dir, test_dir, models_dir, 
                          output_dir, num_folds=5, batch_size=16, 
                          num_workers=0, num_classes=2, device='cuda'):
    """
    评估单个模型的所有fold
    
    参数:
        model_name (str): 模型名称
        cv_data_dir (str): 交叉验证数据目录
        test_dir (str): 测试集目录
        models_dir (str): 模型权重目录
        output_dir (str): 输出目录
        num_folds (int): 折数
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        num_classes (int): 分类数量
        device (str): 设备
        
    返回:
        pd.DataFrame: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")
    
    # 创建模型结果保存目录
    model_results_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    # 创建ROC曲线保存目录
    roc_dir = os.path.join(model_results_dir, 'roc_curves')
    os.makedirs(roc_dir, exist_ok=True)
    
    # 存储所有fold的结果
    all_results = []
    
    for fold in range(1, num_folds + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")
        
        # 创建fold结果保存目录
        fold_results_dir = os.path.join(model_results_dir, f'fold_{fold}')
        os.makedirs(fold_results_dir, exist_ok=True)
        
        # 设置数据路径
        train_dir = os.path.join(cv_data_dir, f'fold_{fold}', 'train')
        val_dir = os.path.join(cv_data_dir, f'fold_{fold}', 'val')
        
        # 检查路径是否存在
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print(f"错误: 数据目录不存在，跳过此折")
            continue
        
        # 计算均值和标准差
        print("计算均值和标准差...")
        mean, std = calculate_mean_std(train_dir, model_name)
        print(f"均值: {mean}, 标准差: {std}")
        
        # 创建数据转换
        if 'Inception' in model_name:
            pad_transform = pad_to_square_299_transform
        else:
            pad_transform = pad_to_square_transform
        
        transform = transforms.Compose([
            transforms.Lambda(pad_transform),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # 加载数据集
        print("加载数据集...")
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        
        # 加载模型
        model_dir = os.path.join(models_dir, model_name)
        
        # 尝试多种可能的模型文件名格式
        possible_names = [
            f'{model_name.lower()}_fold_{fold}.pth',
            f'{model_name}_fold_{fold}.pth',
            f'{model_name.lower()}_fold{fold}.pth',
            f'{model_name}_fold{fold}.pth',
            f'fold_{fold}.pth',
            f'fold{fold}.pth',
            f'model_fold_{fold}.pth'
        ]
        
        model_path = None
        for name in possible_names:
            temp_path = os.path.join(model_dir, name)
            if os.path.exists(temp_path):
                model_path = temp_path
                break
        
        if model_path is None:
            print(f"错误: 无法找到 Fold {fold} 的模型文件，跳过此折")
            continue
        
        print(f"加载模型权重: {model_path}")
        model = create_model(model_name, num_classes=num_classes)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"错误: 加载模型权重失败: {e}")
            traceback.print_exc()
            continue
        
        # 评估训练集、验证集、测试集
        train_metrics = predict_and_save_probabilities(
            model, train_loader, 'train', fold_results_dir, device
        )
        val_metrics = predict_and_save_probabilities(
            model, val_loader, 'val', fold_results_dir, device
        )
        test_metrics = predict_and_save_probabilities(
            model, test_loader, 'test', fold_results_dir, device
        )
        
        # 打印结果
        print(f"\n{model_name} Fold {fold} 结果:")
        print_metrics(train_metrics, "训练集")
        print_metrics(val_metrics, "验证集")
        print_metrics(test_metrics, "测试集")
        
        # 绘制ROC曲线
        train_probs_df = pd.read_csv(os.path.join(fold_results_dir, 'train_probabilities.csv'))
        val_probs_df = pd.read_csv(os.path.join(fold_results_dir, 'val_probabilities.csv'))
        test_probs_df = pd.read_csv(os.path.join(fold_results_dir, 'test_probabilities.csv'))
        
        roc_save_path = os.path.join(roc_dir, f'roc_curve_fold_{fold}.png')
        plot_roc_curve(
            train_probs_df['prob_malignant'].values, train_probs_df['true_label'].values,
            val_probs_df['prob_malignant'].values, val_probs_df['true_label'].values,
            test_probs_df['prob_malignant'].values, test_probs_df['true_label'].values,
            roc_save_path
        )
        
        # 保存此fold的结果
        fold_result = {
            'model': model_name,
            'fold': fold,
            # 训练集指标
            'train_accuracy': train_metrics['accuracy'],
            'train_balanced_accuracy': train_metrics['balanced_accuracy'],
            'train_sensitivity': train_metrics['sensitivity'],
            'train_specificity': train_metrics['specificity'],
            'train_precision_benign': train_metrics['precision_benign'],
            'train_recall_benign': train_metrics['recall_benign'],
            'train_f1_benign': train_metrics['f1_benign'],
            'train_precision_malignant': train_metrics['precision_malignant'],
            'train_recall_malignant': train_metrics['recall_malignant'],
            'train_f1_malignant': train_metrics['f1_malignant'],
            'train_mcc': train_metrics['mcc'],
            'train_auc': train_metrics['auc'],
            'train_benign_acc': train_metrics['benign_acc'],
            'train_malignant_acc': train_metrics['malignant_acc'],
            # 验证集指标
            'val_accuracy': val_metrics['accuracy'],
            'val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'val_sensitivity': val_metrics['sensitivity'],
            'val_specificity': val_metrics['specificity'],
            'val_precision_benign': val_metrics['precision_benign'],
            'val_recall_benign': val_metrics['recall_benign'],
            'val_f1_benign': val_metrics['f1_benign'],
            'val_precision_malignant': val_metrics['precision_malignant'],
            'val_recall_malignant': val_metrics['recall_malignant'],
            'val_f1_malignant': val_metrics['f1_malignant'],
            'val_mcc': val_metrics['mcc'],
            'val_auc': val_metrics['auc'],
            'val_benign_acc': val_metrics['benign_acc'],
            'val_malignant_acc': val_metrics['malignant_acc'],
            # 测试集指标
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_sensitivity': test_metrics['sensitivity'],
            'test_specificity': test_metrics['specificity'],
            'test_precision_benign': test_metrics['precision_benign'],
            'test_recall_benign': test_metrics['recall_benign'],
            'test_f1_benign': test_metrics['f1_benign'],
            'test_precision_malignant': test_metrics['precision_malignant'],
            'test_recall_malignant': test_metrics['recall_malignant'],
            'test_f1_malignant': test_metrics['f1_malignant'],
            'test_mcc': test_metrics['mcc'],
            'test_auc': test_metrics['auc'],
            'test_benign_acc': test_metrics['benign_acc'],
            'test_malignant_acc': test_metrics['malignant_acc']
        }
        
        all_results.append(fold_result)
    
    # 如果没有成功评估任何fold，返回None
    if not all_results:
        print(f"警告: {model_name} 没有成功评估任何fold")
        return None
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 计算平均值
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    avg_result = results_df[numeric_cols].mean().to_dict()
    avg_result['model'] = model_name
    avg_result['fold'] = 'Average'
    
    # 添加平均值行
    avg_df = pd.DataFrame([avg_result])
    results_df = pd.concat([results_df, avg_df], ignore_index=True)
    
    # 保存详细结果
    detailed_results_path = os.path.join(model_results_dir, f'{model_name}_detailed_results.csv')
    results_df.to_csv(detailed_results_path, index=False)
    print(f"\n{model_name} 详细结果已保存到 {detailed_results_path}")
    
    # 保存交叉验证汇总结果
    cv_results_path = os.path.join(model_results_dir, f'{model_name}_cross_validation_results.csv')
    avg_df.to_csv(cv_results_path, index=False)
    print(f"{model_name} 交叉验证结果已保存到 {cv_results_path}")
    
    return results_df


def evaluate_all_models(model_list, cv_data_dir, test_dir, models_dir, 
                       output_dir, num_folds=5, batch_size=16, num_workers=0,
                       num_classes=2, device='cuda', save_summary=True):
    """
    评估所有模型
    
    参数:
        model_list (list): 模型名称列表（如果为None，则自动检测models_dir中的所有模型）
        cv_data_dir (str): 交叉验证数据目录
        test_dir (str): 测试集目录
        models_dir (str): 模型权重目录
        output_dir (str): 输出目录
        num_folds (int): 折数
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        num_classes (int): 分类数量
        device (str): 设备
        save_summary (bool): 是否保存汇总结果
        
    返回:
        dict: 评估结果统计
    """
    print(f"\n{'='*60}")
    print(f"开始评估所有模型")
    print(f"{'='*60}")
    
    # 创建结果保存目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定模型列表，自动检测
    if model_list is None:
        if not os.path.exists(models_dir):
            raise ValueError(f"模型目录不存在: {models_dir}")
        
        model_list = [d for d in os.listdir(models_dir)
                     if os.path.isdir(os.path.join(models_dir, d))]
        
        if not model_list:
            raise ValueError(f"在 {models_dir} 中没有找到任何模型文件夹")
    
    print(f"\n找到 {len(model_list)} 个模型:")
    for model_name in model_list:
        print(f"  - {model_name}")
    
    # 评估所有模型
    all_model_results = []
    start_time = time.time()
    
    for model_name in model_list:
        try:
            results_df = evaluate_single_model(
                model_name=model_name,
                cv_data_dir=cv_data_dir,
                test_dir=test_dir,
                models_dir=models_dir,
                output_dir=output_dir,
                num_folds=num_folds,
                batch_size=batch_size,
                num_workers=num_workers,
                num_classes=num_classes,
                device=device
            )
            
            if results_df is not None:
                # 提取平均结果
                avg_result = results_df[results_df['fold'] == 'Average'].iloc[0].to_dict()
                all_model_results.append(avg_result)
                
        except Exception as e:
            print(f"评估 {model_name} 时出错: {e}")
            traceback.print_exc()
            continue
    
    total_time = (time.time() - start_time) / 60
    
    # 如果没有成功评估任何模型
    if not all_model_results:
        print("错误: 没有成功评估任何模型")
        return None
    
    # 保存汇总结果
    if save_summary:
        _save_evaluation_summary(all_model_results, output_dir)
    
    print(f"\n总运行时间: {total_time:.2f} 分钟")
    
    return {
        'total_time': total_time,
        'num_models': len(all_model_results),
        'results': all_model_results,
        'output_dir': output_dir
    }


def _save_evaluation_summary(all_model_results, output_dir):
    """保存评估汇总结果"""
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False
        print("提示: 安装 tabulate 可以获得更好的表格显示效果")
    
    # 创建汇总DataFrame
    summary_df = pd.DataFrame(all_model_results)
    
    # 只保留测试集的指标
    test_cols = ['model'] + [col for col in summary_df.columns if col.startswith('test_')]
    summary_df_test = summary_df[test_cols]
    
    # 按MCC排序
    summary_df_test = summary_df_test.sort_values('test_mcc', ascending=False)
    
    # 保存汇总结果
    summary_path = os.path.join(output_dir, 'all_models_summary.csv')
    summary_df_test.to_csv(summary_path, index=False)
    print(f"\n所有模型的汇总结果已保存到 {summary_path}")
    
    # 打印汇总表格
    if has_tabulate:
        print("\n所有模型的测试集性能指标汇总:")
        print(tabulate(summary_df_test, headers='keys', tablefmt='psql', 
                      showindex=False, floatfmt='.4f'))
    
    # 绘制性能热图
    _plot_performance_heatmap(summary_df_test, output_dir)
    
    # Top4模型分析
    _analyze_top_models(summary_df, summary_df_test, output_dir, top_n=4)


def _plot_performance_heatmap(summary_df_test, output_dir):
    """绘制性能热图"""
    plt.figure(figsize=(16, 10))
    
    # 选择要展示的指标
    metrics_to_plot = [
        'test_precision_benign', 'test_recall_benign', 'test_f1_benign',
        'test_precision_malignant', 'test_recall_malignant', 'test_f1_malignant',
        'test_balanced_accuracy', 'test_sensitivity', 'test_specificity',
        'test_mcc', 'test_auc'
    ]
    
    # 创建热图数据
    heatmap_data = summary_df_test[['model'] + metrics_to_plot].set_index('model')
    
    # 绘制热图
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title('Model Performance Heatmap (Test Set)', fontsize=16, pad=20)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'all_models_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热图已保存到 {heatmap_path}")


def _analyze_top_models(summary_df, summary_df_test, output_dir, top_n=4):
    """分析Top N模型，并找出每个模型中test_mcc最高的那一折作为代表性能"""
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False
        print("提示: 安装 tabulate 可以获得更好的表格显示效果")

    # 找出MCC值前N的模型
    top_models = summary_df_test.nlargest(top_n, 'test_mcc')
    top_model_names = top_models['model'].tolist()
    
    print(f"\nMCC值前{top_n}的模型: {top_model_names}")
    
    # 存储每个模型最佳折的性能数据
    best_folds_data = []
    
    for model_name in top_model_names:
        # 构建模型详细结果文件路径
        model_eval_folder = os.path.join(output_dir, model_name)
        detailed_results_file = f"{model_name}_detailed_results.csv"
        detailed_results_path = os.path.join(model_eval_folder, detailed_results_file)
        
        if not os.path.exists(detailed_results_path):
            print(f"警告: 找不到详细结果文件 {detailed_results_path}，跳过模型 {model_name}")
            continue
        
        # 读取详细结果文件
        detailed_df = pd.read_csv(detailed_results_path)
        
        # 排除'Average'行，只考虑实际的折
        fold_df = detailed_df[detailed_df['fold'] != 'Average']
        
        if fold_df.empty:
            print(f"警告: {model_name} 的详细结果不包含任何折数据")
            continue
        
        # 找到test_mcc值最大的折数
        max_mcc_idx = fold_df['test_mcc'].idxmax()
        best_fold = fold_df.loc[max_mcc_idx, 'fold']
        best_mcc = fold_df.loc[max_mcc_idx, 'test_mcc']
        
        # 获取最佳折的所有性能指标
        best_fold_data = fold_df.loc[max_mcc_idx].to_dict()
        best_folds_data.append(best_fold_data)
        
        print(f"模型 {model_name}: 最佳折数 {best_fold}, MCC值 {best_mcc:.4f}")
    
    # 创建最佳折性能的DataFrame
    if not best_folds_data:
        print("警告: 无法找到任何模型的最佳折数据")
        return
    
    best_folds_df = pd.DataFrame(best_folds_data)
    
    # 选择关键指标
    key_metrics = ['model', 'fold',
                   'train_sensitivity', 'train_specificity', 'train_auc',
                   'val_sensitivity', 'val_specificity', 'val_auc',
                   'test_sensitivity', 'test_specificity', 'test_auc']
    
    # 确保所有关键指标都存在
    available_metrics = [m for m in key_metrics if m in best_folds_df.columns]
    top_summary = best_folds_df[available_metrics]
    
    # 保存最佳折的性能指标
    top_path = os.path.join(output_dir, f'top{top_n}_models_summary.csv')
    top_summary.to_csv(top_path, index=False)
    print(f"\nMCC值前{top_n}的模型最佳折汇总结果已保存到 {top_path}")
    
    if has_tabulate:
        print(f"\nMCC值前{top_n}的模型最佳折性能指标汇总:")
        print(tabulate(top_summary, headers='keys', tablefmt='psql',
                      showindex=False, floatfmt='.4f'))
    
    # 绘制Top N模型最佳折的热图
    plt.figure(figsize=(12, 6))
    metrics_to_plot = [
        'test_precision_benign', 'test_recall_benign', 'test_f1_benign',
        'test_precision_malignant', 'test_recall_malignant', 'test_f1_malignant',
        'test_balanced_accuracy', 'test_sensitivity', 'test_specificity',
        'test_mcc', 'test_auc'
    ]
    
    # 确保所有要绘制的指标都存在
    available_plot_metrics = [m for m in metrics_to_plot if m in best_folds_df.columns]
    
    # 设置模型名称为索引
    heatmap_df = best_folds_df.set_index('model')
    top_heatmap_data = heatmap_df[available_plot_metrics]
    
    sns.heatmap(top_heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title(f'Top {top_n} Models Best Fold Performance Heatmap (Test Set)', fontsize=14, pad=15)
    plt.xlabel('Metrics', fontsize=11)
    plt.ylabel('Models', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    top_heatmap_path = os.path.join(output_dir, f'top{top_n}_models_heatmap.png')
    plt.savefig(top_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"前{top_n}名模型最佳折热图已保存到 {top_heatmap_path}")
