#!/usr/bin/env python
"""
使用训练好的DenseNet161模型对新数据进行推理并为恶性预测添加置信度标记
- 预测判断：torch.max（等价于固定阈值0.5），二分类标准做法
- 文件名置信度：保留4位小数原始数值
- CONFIDENCE_THRESHOLD：只用于筛选哪些恶性图片值得复制
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import shutil
import argparse

from shilab_classifier.models.model_definitions import create_model
from shilab_classifier.training.preprocessing import pad_to_square_transform, pad_to_square_299_transform


# ============================================================
# 数据加载类
# ============================================================
class UnlabeledImageDataset(Dataset):
    """处理单文件夹结构的无标签图像数据集"""

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform   = transform
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(exts)
        ])
        self.samples = [(os.path.join(folder_path, f), 0) for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.samples[idx][0]
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0


# ============================================================
# 模型权重加载（兼容多种保存格式）
# ============================================================
def load_model_weights(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("已加载 checkpoint['model_state_dict']")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("已加载 checkpoint['state_dict']")
    else:
        model.load_state_dict(checkpoint)
        print("已加载直接保存的 state_dict")
    return model


# ============================================================
# 处理恶性预测图片
# ============================================================
def process_malignant_predictions(results_df, malignant_output_dir, confidence_threshold, output_dir):
    """
    复制恶性预测图片到输出目录，文件名附带4位小数置信度
    筛选条件：predicted_class == 'malignant' 且 prob_malignant >= confidence_threshold
    """
    malignant_predictions = results_df[
        (results_df['predicted_class'] == 'malignant') &
        (results_df['prob_malignant'] >= confidence_threshold)
    ]

    print(f"找到 {len(malignant_predictions)} 张恶性预测图片（置信度 ≥ {confidence_threshold}）")

    if len(malignant_predictions) == 0:
        print("没有找到符合条件的恶性预测图片")
        return

    processed_count    = 0
    renamed_files_info = []

    for _, row in malignant_predictions.iterrows():
        original_path = row['image_path']
        confidence    = row['prob_malignant']

        name, ext = os.path.splitext(os.path.basename(original_path))

        # ✅ 保留4位小数原始数值，格式统一且可读
        # 示例：0.9521 → "cell_001_malignant_0.9521.jpg"
        #       1.0000 → "cell_002_malignant_1.0000.jpg"
        confidence_str = f"{confidence:.4f}"
        new_filename   = f"{name}_malignant_{confidence_str}{ext}"
        new_path       = os.path.join(malignant_output_dir, new_filename)

        try:
            shutil.copy2(original_path, new_path)
            processed_count += 1
            renamed_files_info.append({
                'original_path': original_path,
                'new_path'     : new_path,
                'confidence'   : confidence,
            })
        except Exception as e:
            print(f"  ⚠️  处理文件 {original_path} 时出错: {e}")

    if renamed_files_info:
        renamed_df   = pd.DataFrame(renamed_files_info)
        renamed_path = os.path.join(output_dir, 'malignant_images_with_confidence.csv')
        renamed_df.to_csv(renamed_path, index=False, encoding='utf-8-sig')
        print(f"已处理 {processed_count} 张恶性图片，详情保存到 {renamed_path}")
        print(f"带置信度的恶性图片已保存到 {malignant_output_dir}")


# ============================================================
# 主函数
# ============================================================
def predict_new_data(
    model_name="DenseNet161",
    model_path=None,
    input_dir=None,
    output_dir=None,
    batch_size=16,
    num_workers=0,
    confidence_threshold=0.5,
    device=None,
    mean=None,
    std=None
):
    """对新收集的数据进行预测并为恶性预测添加置信度标记"""
    
    # 参数检查
    if model_path is None or input_dir is None or output_dir is None:
        raise ValueError("必须提供模型路径、输入目录和输出目录")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    num_classes = 2
    idx_to_class = {0: "benign", 1: "malignant"}
    
    # 标准化参数（如果未提供，则使用ImageNet默认值）
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet默认值
    
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet默认值
    
    # 创建输出目录
    malignant_output_dir = os.path.join(output_dir, 'malignant_images')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(malignant_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"使用 {model_name} 模型预测新数据")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"置信度阈值: {confidence_threshold}")

    # ----------------------------------------------------------
    # 1. 加载模型
    # ----------------------------------------------------------
    print(f"加载模型权重: {model_path}")
    model = create_model(model_name, num_classes=num_classes)
    model = load_model_weights(model, model_path, device)
    model = model.to(device)
    model.eval()

    # ----------------------------------------------------------
    # 2. 数据预处理（必须和训练时完全一致）
    # ----------------------------------------------------------
    print(f"使用标准化参数 - 均值: {mean}, 标准差: {std}")
    
    # 选择合适的填充变换
    if 'Inception' in model_name:
        pad_transform = pad_to_square_299_transform
    else:
        pad_transform = pad_to_square_transform
        
    transform = transforms.Compose([
        transforms.Lambda(pad_transform),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # ----------------------------------------------------------
    # 3. 加载数据
    # ----------------------------------------------------------
    print(f"加载新数据: {input_dir}")

    # 过滤隐藏文件夹（如 .ipynb_checkpoints）再判断目录结构
    has_subfolders = any(
        os.path.isdir(os.path.join(input_dir, d))
        for d in os.listdir(input_dir)
        if not d.startswith('.')
    )

    if has_subfolders:
        print("检测到子文件夹结构，使用ImageFolder加载数据...")
        print("⚠️  注意：请确认下方打印的类别映射和训练时一致！")
        try:
            dataset    = datasets.ImageFolder(input_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
            print(f"找到 {len(dataset)} 张图片，分为 {len(dataset.class_to_idx)} 个类别")
            for class_name, idx in dataset.class_to_idx.items():
                print(f"  - 文件夹类别 {idx}: {class_name}")
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    else:
        print("检测到单一文件夹结构，使用自定义数据集加载...")
        dataset    = UnlabeledImageDataset(input_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        print(f"找到 {len(dataset)} 张图片")

    if len(dataset) == 0:
        print("错误：没有找到任何图片，请检查路径和文件格式。")
        return None

    # ----------------------------------------------------------
    # 4. 推理
    # ----------------------------------------------------------
    print("\n开始预测...")
    all_paths  = []
    all_logits = []
    all_probs  = []
    all_preds  = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  预测批次 {batch_idx + 1}/{len(dataloader)}")

            # 获取当前批次图像路径
            batch_size_actual = inputs.size(0)
            start_idx   = batch_idx * batch_size
            batch_paths = [
                dataset.samples[i][0]
                for i in range(start_idx, min(start_idx + batch_size_actual, len(dataset)))
            ]

            inputs  = inputs.to(device)
            outputs = model(inputs)

            # 兼容 Inception 等多输出模型
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_paths.extend(batch_paths)
            all_logits.append(outputs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    # ----------------------------------------------------------
    # 5. 整理结果
    # ----------------------------------------------------------
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_preds  = np.concatenate(all_preds,  axis=0)

    results_df = pd.DataFrame({
        'image_path'     : all_paths,
        'filename'       : [os.path.basename(p) for p in all_paths],
        'predicted_idx'  : all_preds,
        'predicted_class': [idx_to_class[int(idx)] for idx in all_preds],
        'logit_benign'   : all_logits[:, 0],
        'logit_malignant': all_logits[:, 1],
        'prob_benign'    : all_probs[:, 0],
        'prob_malignant' : all_probs[:, 1],
    })

    results_path = os.path.join(output_dir, 'prediction_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到 {results_path}")

    # 统计
    class_counts = results_df['predicted_class'].value_counts()
    print("\n预测类别统计:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} 张 ({count / len(results_df) * 100:.1f}%)")

    # ----------------------------------------------------------
    # 6. 处理恶性图片
    # ----------------------------------------------------------
    print("\n开始处理恶性预测图片...")
    process_malignant_predictions(results_df, malignant_output_dir, confidence_threshold, output_dir)

    # ----------------------------------------------------------
    # 7. 绘制概率分布图
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['prob_malignant'], bins=20, alpha=0.7,
             color='steelblue', edgecolor='white')
    plt.axvline(x=confidence_threshold, color='red', linestyle='--',
                label=f'Threshold = {confidence_threshold}')
    plt.title('Malignant Prediction Probability Distribution')
    plt.xlabel('Malignant probability')
    plt.ylabel('Number of images')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    prob_hist_path = os.path.join(output_dir, 'malignant_probability_distribution.png')
    plt.savefig(prob_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测概率分布图已保存到 {prob_hist_path}")

    print("\n✅ 预测完成！")
    return results_df


# ============================================================
# 命令行参数解析
# ============================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='使用训练好的模型对新数据进行推理')
    
    parser.add_argument('-m', '--model', type=str, default="DenseNet161",
                        help='模型名称 (默认: DenseNet161)')
    
    parser.add_argument('-w', '--weights', type=str, required=True,
                        help='模型权重文件路径 (.pth)')
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入数据目录路径')
    
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='输出结果目录路径')
    
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='批处理大小 (默认: 16)')
    
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='数据加载线程数 (默认: 0)')
    
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='恶性预测置信度阈值 (默认: 0.5)')
    
    parser.add_argument('--mean', type=float, nargs=3, default=None,
                        help='标准化均值 [R G B] (默认: ImageNet均值 [0.485, 0.456, 0.406])')
    
    parser.add_argument('--std', type=float, nargs=3, default=None,
                        help='标准化标准差 [R G B] (默认: ImageNet标准差 [0.229, 0.224, 0.225])')
    
    return parser.parse_args(argv)

def main(argv=None):
    """binary_infer 的命令行入口。成功返回 0，失败时向调用方抛出异常。"""
    args = parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    predict_new_data(
        model_name=args.model,
        model_path=args.weights,
        input_dir=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
        confidence_threshold=args.threshold,
        device=device,
        mean=args.mean,
        std=args.std,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())