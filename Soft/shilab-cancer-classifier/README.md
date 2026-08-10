# ShiLab Binary Classifier

ShiLab课题组的二分类模型工具包

## 功能特性

- ✅ 数据集分割（训练集、验证集、测试集）
- 🚧 模型训练（开发中）
- 🚧 模型评估（开发中）
- 🚧 模型推理（开发中）

## Step-4 后的多癌种推理

`shilab-cancer-infer` 只读取假阳性去除聚类后、名称为
`matched_malignant_cells` 的文件夹；可传入单个患者文件夹，也可传入包含
多个患者输出的 step-4 根目录。它不会混入 `rule1_matched_cells` 等中间结果。

推理时必须显式提供**训练时 ImageFolder 的类别顺序**和 mean/std；不需要
额外的 metadata 文件：

```bash
python -m shilab_cancer_classifier.infer.cancer_infer \
  --model DenseNet161 --weights /path/to/model.pth \
  --input /path/to/step4_output --output /path/to/cancer_infer \
  --classes Bile_duct Breast Cervix Colorectum Endometrium Esophagus Gastric Mesothelioma NSCLC Ovary Pancreas SCLC \
  --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225
```

输出包括逐细胞 `prediction_results.csv`、逐患者 `patient_summary.csv`、按预测癌种
分类的高置信度图片，以及各癌种概率分布图。

## 安装

### 从源码安装

```bash
git clone https://github.com/shilab/binary-classifier.git
cd shilab-binary-classifier
pip install -e .
```

### 使用pip安装（发布后）

pip install shilab-binary-classifier

## 快速开始

### 数据集分割

```
from shilab_cancer_classifier import split_dataset

# 分割数据集
result = split_dataset(
    benign_path="path/to/benign/images",
    malignant_path="path/to/malignant/images",
    output_dir="path/to/output",
    split_ratio=0.9,  # 90%用于训练验证集，10%用于测试集
    random_seed=42
)

print(f"良性图像: {result['benign']}")
print(f"恶性图像: {result['malignant']}")
print(f"日志保存在: {result['log_path']}")

```

## 输出结构

output_dir/
├── train_val/
│   ├── benign/
│   └── malignant/
├── test/
│   ├── benign/
│   └── malignant/
└── dataset_split_log.xlsx



### 4. 评估模型

```python
import torch
from shilab_cancer_classifier import evaluate_all_models

# 评估所有训练好的模型
result = evaluate_all_models(
    model_list=None,  # None表示自动检测所有模型
    cv_data_dir="path/to/cross_validation_data",
    test_dir="path/to/test_data",
    models_dir="path/to/trained_models",
    output_dir="path/to/evaluation_results",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"评估完成！总用时: {result['total_time']:.2f} 分钟")
print(f"评估了 {result['num_models']} 个模型")
```

## 评估输出

```
evaluation_results/
├── ModelName1/
│   ├── fold_1/
│   │   ├── train_probabilities.csv
│   │   ├── train_metrics.csv
│   │   ├── train_confusion_matrix.png
│   │   ├── val_probabilities.csv
│   │   ├── val_metrics.csv
│   │   ├── val_confusion_matrix.png
│   │   ├── test_probabilities.csv
│   │   ├── test_metrics.csv
│   │   └── test_confusion_matrix.png
│   ├── fold_2/
│   │   └── ...
│   ├── roc_curves/
│   │   ├── roc_curve_fold_1.png
│   │   └── ...
│   ├── ModelName1_detailed_results.csv
│   └── ModelName1_cross_validation_results.csv
├── ModelName2/
│   └── ...
├── all_models_summary.csv
├── all_models_heatmap.png
├── top4_models_summary.csv
└── top4_models_heatmap.png
```
