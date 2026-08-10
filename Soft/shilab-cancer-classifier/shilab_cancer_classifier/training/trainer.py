"""
19个SOTA模型训练模块
"""
import os
import time
import math
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import sys
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from shilab_cancer_classifier.models.model_definitions import create_model
from shilab_cancer_classifier.config import CANCER_CLASSES, NUM_CLASSES
from shilab_cancer_classifier.training.preprocessing import (
    pad_to_square_transform,
    pad_to_square_299_transform,
    calculate_mean_std
)

def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_lr_schedule(optimizer, epoch, num_epochs, initial_lr):
    """
    余弦学习率调度
    严格按照原代码实现
    """
    lr = initial_lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                num_epochs,
                initial_lr,
                patience,
                device,
                class_names=None,
                log_file=None
                ):
    """
    训练模型
    
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        initial_lr: 初始学习率
        patience: 早停耐心值
        device: 训练设备
        log_file: 日志文件
        
    返回:
        tuple: (model, history, best_val_acc, best_train_acc, best_detail_matrix)
    """
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_model_wts = None
    class_names = list(class_names or CANCER_CLASSES)
    labels_for_cm = list(range(len(class_names)))
    best_detail_matrix = None  # 用于保存最佳模型的 detail_matrix
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_class_acc': [], 'val_class_acc': [],
        'learning_rate': []  # 添加学习率跟踪
    }
    
    # 添加早停机制 - 修改为监控验证集loss
    patience = patience  # 10步耐心早停
    counter = 0
    best_val_loss = float('inf')  # 初始化为无穷大
    
    def log_message(message):
        print(message)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        log_message(f"Epoch {epoch+1}/{num_epochs} 开始时间: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        # 更新学习率
        current_lr = cosine_lr_schedule(optimizer, epoch, num_epochs, initial_lr)
        log_message(f"当前学习率: {current_lr:.8f}")
        history['learning_rate'].append(current_lr)
        
        # 训练阶段
        model.train()
        log_message("开始训练阶段...")
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        
        # 尝试迭代训练数据
        try:
            log_message(f"训练数据批次数: {len(train_loader)}")
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                #if batch_idx % 10 == 0:
                #    log_message(f"处理训练批次 {batch_idx}/{len(train_loader)}")
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                # 前向传播
                if hasattr(outputs, 'logits'):
                    # GoogLeNet输出
                    logits = outputs.logits
                elif isinstance(outputs, tuple) and len(outputs) > 1:
                    # InceptionV3输出 (通常是一个元组，主输出在第一个位置)
                    logits = outputs[0]
                else:
                    # 常规模型输出
                    logits = outputs
                # 在评估模式下，模型输出已经是softmax后的结果，需要取对数以便与CrossEntropyLoss兼容
                loss = criterion(logits, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
            log_message("训练阶段完成!")
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
        except Exception as e:
            error_msg = f"训练阶段出错: {e}"
            log_message(error_msg)
            traceback.print_exc()
            
            # 将错误信息保存到单独的错误日志文件
            if log_file:
                error_file = log_file.replace('.log', '_error.log')
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"训练阶段出错: {e}\n")
                    f.write(traceback.format_exc())
            
            return model, history, 0, 0, {}
        
        # 验证阶段
        model.eval()
        log_message("开始验证阶段...")
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        try:
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    #if batch_idx % 10 == 0:
                    #    log_message(f"处理验证批次 {batch_idx}/{len(val_loader)}")
                    if inputs.shape[0] == 1:
                        # print("检测到单样本 Batch，正在执行复制补齐...")
                        # 在第0维（batch维）拼接自己，变成 [2, C, H, W]
                        inputs = torch.cat([inputs, inputs], dim=0)
                        labels = torch.cat([labels, labels], dim=0)  # 修正：使用 labels 而不是 targets
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # 前向传播
                    outputs = model(inputs)
                    # 前向传播
                    if hasattr(outputs, 'logits'):
                        # GoogLeNet输出
                        logits = outputs.logits
                    elif isinstance(outputs, tuple) and len(outputs) > 1:
                        # InceptionV3输出 (通常是一个元组，主输出在第一个位置)
                        logits = outputs[0]
                    else:
                        # 常规模型输出
                        logits = outputs
                    # 在评估模式下，模型输出已经是softmax后的结果，需要取对数以便与CrossEntropyLoss兼容
                    loss = criterion(logits, labels)
                        
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            log_message("验证阶段完成!")
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
        except Exception as e:
            error_msg = f"验证阶段出错: {e}"
            log_message(error_msg)
            traceback.print_exc()
            
            # 将错误信息保存到单独的错误日志文件
            if log_file:
                error_file = log_file.replace('.log', '_error.log')
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"验证阶段出错: {e}\n")
                    f.write(traceback.format_exc())
                    
            return model, history, 0, 0, {}
        
        # 计算混淆矩阵并更新 detail_matrix
        try:
            train_cm = confusion_matrix(train_labels, train_preds, labels=labels_for_cm)
            val_cm = confusion_matrix(val_labels, val_preds, labels=labels_for_cm)

            def per_class_detail(cm, prefix):
                detail = {}
                class_acc = {}
                for idx, class_name in enumerate(class_names):
                    total = cm[idx, :].sum()
                    correct = cm[idx, idx]
                    acc = correct / total if total > 0 else 0.0
                    safe_name = class_name.lower()
                    detail[f'{prefix}_{safe_name}_correct'] = int(correct)
                    detail[f'{prefix}_{safe_name}_total'] = int(total)
                    detail[f'{prefix}_{safe_name}_acc'] = float(acc)
                    class_acc[class_name] = float(acc)
                detail[f'{prefix}_class_acc'] = class_acc
                return detail

            detail_matrix = {}
            detail_matrix.update(per_class_detail(train_cm, 'train'))
            detail_matrix.update(per_class_detail(val_cm, 'val'))
            detail_matrix['train_confusion_matrix'] = train_cm
            detail_matrix['val_confusion_matrix'] = val_cm
        except Exception as e:
            error_msg = f"计算混淆矩阵出错: {e}"
            log_message(error_msg)
            traceback.print_exc()
            
            # 将错误信息保存到单独的错误日志文件
            if log_file:
                error_file = log_file.replace('.log', '_error.log')
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"计算混淆矩阵出错: {e}\n")
                    f.write(traceback.format_exc())
                    
            detail_matrix = {
                'train_class_acc': {class_name: 0.0 for class_name in class_names},
                'val_class_acc': {class_name: 0.0 for class_name in class_names},
            }

        # 保存最佳模型 - 仍然跟踪最佳验证准确率(用于返回)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_model_wts = model.state_dict().copy()
            best_detail_matrix = detail_matrix.copy()
        
        # 早停机制 - 修改为监控验证集loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # 重置早停计数器
        else:
            counter += 1  # 增加早停计数器
            
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_class_acc'].append(detail_matrix['train_class_acc'])
        history['val_class_acc'].append(detail_matrix['val_class_acc'])
        
        epoch_time = time.time() - start_time
        log_message(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'LR: {current_lr:.8f} | Time: {epoch_time:.2f}s')
    
        # 检查是否早停
        if counter >= patience:
            log_message(f'Early stopping triggered after {epoch+1} epochs (validation loss not improved for {patience} epochs)')
            break
    
    # 加载最佳模型权重
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    return model, history, best_val_acc, best_train_acc, best_detail_matrix


def plot_training_curves(history, model_name, fold, save_dir):
    """
    绘制训练曲线
    
    参数:
        history (dict): 训练历史记录
        model_name (str): 模型名称
        fold (int): 折数
        save_dir (str): 保存目录
    """
    # 创建图表目录
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 绘制损失函数曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Fold {fold} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower()}_fold_{fold}_loss.png'))
    plt.close()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} - Fold {fold} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower()}_fold_{fold}_accuracy.png'))
    plt.close()
    
    # 绘制平衡准确率曲线 (Balanced Accuracy)
    plt.figure(figsize=(10, 5))
    # 计算平衡准确率（各类别召回率的平均）
    train_balanced_acc = [
        float(np.mean(list(epoch_acc.values()))) if epoch_acc else 0.0
        for epoch_acc in history.get('train_class_acc', [])
    ]
    val_balanced_acc = [
        float(np.mean(list(epoch_acc.values()))) if epoch_acc else 0.0
        for epoch_acc in history.get('val_class_acc', [])
    ]
    
    plt.plot(train_balanced_acc, label='Train Balanced Acc')
    plt.plot(val_balanced_acc, label='Val Balanced Acc')
    plt.title(f'{model_name} - Fold {fold} Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower()}_fold_{fold}_balanced_acc.png'))
    plt.close()


def train_single_model_fold(   
    model_name, 
    fold, 
    train_dir, 
    val_dir, 
    save_dir, 
    num_epochs=50, 
    batch_size=16, 
    initial_lr=0.001, 
    device='cuda',
    num_classes=NUM_CLASSES,
    num_workers=0,
    patience=10,
    seed=42):  # ✅ 新增 seed 参数，方便统一管理
    """
    训练单个模型的单个fold
    
    参数:
        model_name (str): 模型名称
        fold (int): 折数
        train_dir (str): 训练数据目录
        val_dir (str): 验证数据目录
        save_dir (str): 模型保存目录
        num_epochs (int): 训练轮数
        batch_size (int): 批次大小
        initial_lr (float): 初始学习率
        device (str): 训练设备
        num_classes (int): 分类数量
        num_workers (int): 数据加载线程数
        patience (int): 早停耐心值
        seed (int): 随机种子，默认42
        
    返回:
        dict: 训练历史记录
    """
    # ✅ 第一步：固定所有全局随机状态
    set_seed(seed)
    
    # 设置日志文件
    log_file = os.path.join(save_dir, f'{model_name.lower()}_fold_{fold}.log')
    
    def log_message(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    log_message(f"\n{'='*60}")
    log_message(f"训练 {model_name} - Fold {fold}")
    log_message(f"随机种子: {seed}")  # ✅ 记录种子到日志，方便以后查
    log_message(f"{'='*60}")
    
    # 1. 计算训练集的均值和标准差
    log_message("计算训练集的均值和标准差...")
    mean, std = calculate_mean_std(train_dir, model_name)
    log_message(f"均值: {mean}")
    log_message(f"标准差: {std}")
    
    # 2. 创建数据转换（添加数据增强）
    if 'Inception' in model_name:
        pad_transform = pad_to_square_299_transform
    else:
        pad_transform = pad_to_square_transform
    
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.Lambda(pad_transform),
        # 添加数据增强
        transforms.RandomRotation(15),  # 随机旋转±15度
        transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 50%概率垂直翻转
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 亮度和对比度微调
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 验证数据不需要增强
    val_transform = transforms.Compose([
        transforms.Lambda(pad_transform),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 3. 加载数据集
    log_message("加载数据集...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    class_names = train_dataset.classes
    if len(class_names) != num_classes:
        raise ValueError(
            f"num_classes={num_classes}, but training data contains {len(class_names)} classes: {class_names}"
        )
    if val_dataset.classes != class_names:
        raise ValueError(
            f"Validation classes differ from training classes: "
            f"train={class_names}, val={val_dataset.classes}"
        )
    # 创建固定种子的 generator，和 set_seed 用同一个 seed
    generator = torch.Generator()
    generator.manual_seed(seed) 
    
    # 固定多进程 worker 的随机状态
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True,
        generator=generator, 
        worker_init_fn=seed_worker
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker
    )
    
    log_message(f"训练集样本数: {len(train_dataset)}")
    log_message(f"验证集样本数: {len(val_dataset)}")
    log_message(f"类别: {train_dataset.classes}")
    log_message(f"class_to_idx: {train_dataset.class_to_idx}")
    
    # 4. 创建模型
    log_message(f"创建模型: {model_name}")
    model = create_model(model_name, num_classes)
    model = model.to(device)
    
    # 5. 定义损失函数和优化器
    ####### 2026.6.12 ####加权损失（修正版）
    train_labels_for_weight = np.array(train_dataset.targets)
    classes = np.arange(num_classes)

    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels_for_weight
    )

    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    log_message(f"类别权重: { {class_names[i]: f'{raw_weights[i]:.4f}' for i in range(num_classes)} }")
    ####### 2026.6.12 ####加权损失（修正版）

    # criterion = nn.CrossEntropyLoss()   ← 原来的，已注释掉，保留备用
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0005)
    
    # 6. 训练模型
    log_message(f"\n开始训练 {num_epochs} 个epoch...")
    start_time = time.time()
    
    # 使用原代码的返回值格式
    model, history, best_val_acc, best_train_acc, best_detail_matrix = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs, initial_lr, patience, device, class_names, log_file
    )
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    # 7. 保存最佳模型
    model_save_path = os.path.join(save_dir, f'{model_name.lower()}_fold_{fold}.pth')
    torch.save(model.state_dict(), model_save_path)
    log_message(f"✅ 模型已保存到: {model_save_path}")
    # 8. 绘制训练曲线
    plot_training_curves(history, model_name, fold, save_dir)
    log_message(f"✅ 训练曲线已保存到: {os.path.join(save_dir, 'plots')}")
    
    # 9. 打印详细结果
    log_message(f"\n{'='*60}")
    log_message(f"{model_name} - Fold {fold} 训练完成!")
    log_message(f"最佳验证准确率: {best_val_acc:.4f}")
    log_message(f"对应的训练准确率: {best_train_acc:.4f}")
    if best_detail_matrix:
        log_message("Per-class recall at best validation accuracy:")
        for class_name in class_names:
            safe_name = class_name.lower()
            train_correct = best_detail_matrix.get(f'train_{safe_name}_correct', 0)
            train_total = best_detail_matrix.get(f'train_{safe_name}_total', 0)
            train_acc = best_detail_matrix.get(f'train_{safe_name}_acc', 0.0)
            val_correct = best_detail_matrix.get(f'val_{safe_name}_correct', 0)
            val_total = best_detail_matrix.get(f'val_{safe_name}_total', 0)
            val_acc = best_detail_matrix.get(f'val_{safe_name}_acc', 0.0)
            log_message(
                f"  {class_name}: train {train_correct}/{train_total} ({train_acc:.4f}), "
                f"val {val_correct}/{val_total} ({val_acc:.4f})"
            )
        log_message(f"详细指标:")
        log_message(f"  训练集:")
        log_message(f"  验证集:")
    log_message(f"训练时间: {training_time:.2f} 分钟")
    log_message(f"{'='*60}")
    
    return history


def train_all_models(model_list, cv_data_dir, output_dir, num_folds, 
                     num_epochs, batch_size, initial_lr, device, num_classes, 
                     num_workers, patience, save_log=True,
                     seed=42):
    """
    训练所有模型
    
    参数:
        model_list (list): 模型名称列表
        cv_data_dir (str): 交叉验证数据目录
        output_dir (str): 输出目录
        num_folds (int): 折数
        num_epochs (int): 训练轮数
        batch_size (int): 批次大小
        initial_lr (float): 初始学习率
        device (str): 训练设备
        num_classes (int): 分类数量
        num_workers (int): 数据加载线程数
        patience (int): 早停耐心值
        save_log (bool): 是否保存日志
        seed (int): 随机种子，默认42
        
    返回:
        dict: 训练结果统计
    """
    print(f"\n{'='*60}")
    print(f"开始训练所有模型")
    print(f"模型数量: {len(model_list)}")
    print(f"折数: {num_folds}")
    print(f"总训练次数: {len(model_list) * num_folds}")
    print(f"随机种子: {seed}")
    print(f"{'='*60}")
    
    # 创建总日志文件
    main_log_file = os.path.join(output_dir, "training_main.log")
    with open(main_log_file, 'w', encoding='utf-8') as f:
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型数量: {len(model_list)}\n")
        f.write(f"折数: {num_folds}\n")
        f.write(f"总训练次数: {len(model_list) * num_folds}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"{'='*60}\n")
    
    total_start_time = time.time()
    training_logs = []
    
    for model_name in model_list:
        model_save_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        for fold in range(1, num_folds + 1):
            train_dir = os.path.join(cv_data_dir, f'fold_{fold}', 'train')
            val_dir = os.path.join(cv_data_dir, f'fold_{fold}', 'val')
            
            if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                error_msg = f"❌ 错误: 数据路径不存在 - {train_dir} 或 {val_dir}"
                print(error_msg)
                with open(main_log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + '\n')
                continue
            
            try:
                fold_start_time = time.time()
                history = train_single_model_fold(
                    model_name=model_name,
                    fold=fold,
                    train_dir=train_dir,
                    val_dir=val_dir,
                    save_dir=model_save_dir,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    initial_lr=initial_lr,
                    device=device,
                    num_classes=num_classes,
                    num_workers=num_workers,
                    patience=patience,
                    seed=seed 
                )
                fold_time = (time.time() - fold_start_time) / 60
                
                # 记录日志
                log_entry = {
                    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    '模型': model_name,
                    '折数': fold,
                    '随机种子': seed, 
                    '最佳验证准确率': max(history['val_acc']) if history['val_acc'] else 0.0,  # 添加检查
                    '对应训练准确率': history['train_acc'][history['val_acc'].index(max(history['val_acc']))] if history['val_acc'] else 0.0,
                    '训练时长(分钟)': f"{fold_time:.2f}"
                }
                training_logs.append(log_entry)
                
                # 添加到主日志
                with open(main_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"完成 {model_name} - Fold {fold} | "
                           f"最佳验证准确率: {log_entry['最佳验证准确率']:.4f} | "
                           f"训练时长: {log_entry['训练时长(分钟)']} 分钟\n")
                
            except Exception as e:
                error_msg = f"❌ 训练 {model_name} Fold {fold} 时出错: {e}"
                print(error_msg)
                traceback.print_exc()
                
                # 将错误信息保存到错误日志
                with open(main_log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + '\n')
                    f.write(traceback.format_exc() + '\n')
                
                continue
    
    total_time = (time.time() - total_start_time) / 60
    
    completion_msg = f"\n{'='*60}\n✅ 所有模型训练完成!\n总训练时间: {total_time:.2f} 分钟\n{'='*60}"
    print(completion_msg)
    with open(main_log_file, 'a', encoding='utf-8') as f:
        f.write(completion_msg + '\n')
    
    # 保存日志
    if save_log and training_logs:
        log_path = os.path.join(output_dir, "training_log.xlsx")
        df = pd.DataFrame(training_logs)
        df.to_excel(log_path, index=False, engine='openpyxl')
        print(f"\n📝 训练日志已保存到: {log_path}")
    
    return {
        'total_time': total_time,
        'training_logs': training_logs,
        'output_dir': output_dir
    }
