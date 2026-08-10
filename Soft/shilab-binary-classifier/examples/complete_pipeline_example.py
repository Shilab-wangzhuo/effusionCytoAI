"""
完整的端到端训练评估流程
从数据集划分到模型评估
日期：2026.01.28
作者: limr
"""

import os
import torch
from datetime import datetime
from shilab_classifier import (
    split_dataset,
    prepare_cross_validation,
    train_all_models,
    evaluate_all_models,
    SUPPORTED_MODELS
)


def print_section(title):
    """打印分隔线"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """主函数：完整的训练评估流程"""
    
    # ========================================================================
    # 第0步：配置参数
    # ========================================================================
    print_section("配置参数")
    
    # 路径配置
    RAW_DATA_DIR = r"H:\0UR_classification_model\raw_data"  # 原始数据目录
    BASE_OUTPUT_DIR = r"H:\0UR_classification_model"  # 基础输出目录    

    # 创建时间戳文件夹（可选，用于区分不同的实验）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_DIR = os.path.join(BASE_OUTPUT_DIR, f"experiment_{timestamp}")
    # EXPERIMENT_DIR = os.path.join(BASE_OUTPUT_DIR, f"experiment_limr")
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    # 各阶段路径
    benign_path = os.path.join(RAW_DATA_DIR, "benign")
    malignant_path = os.path.join(RAW_DATA_DIR, "malignant")
    SPLIT_DATA_DIR = os.path.join(EXPERIMENT_DIR, "split_data")
    CV_DATA_DIR = os.path.join(EXPERIMENT_DIR, "cross_validation_data")
    MODELS_DIR = os.path.join(EXPERIMENT_DIR, "train_val_models")
    RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "evaluation_results")
    
    # 使用第1步划分的训练集和验证集
    TRAIN_VAL_DIR = os.path.join(SPLIT_DATA_DIR, 'train_val')    
    TEST_DIR = os.path.join(SPLIT_DATA_DIR, 'test')

    # 训练参数
    NUM_FOLDS = 5
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    INITIAL_LR = 0.001
    PATIENCE = 10
    NUM_WORKERS = 0
    NUM_CLASSES = 2
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 选择要训练的模型（可以选择部分或全部）    
    MODELS_TO_TRAIN = [
        'ResNet50'
    ]
    # 或者训练所有模型：
    # MODELS_TO_TRAIN = SUPPORTED_MODELS       
    
     # 检查原始数据目录是否存在
    if not os.path.exists(RAW_DATA_DIR):
        print(f"\n❌ 错误: 原始数据目录不存在: {RAW_DATA_DIR}")
        return
    
    if not os.path.exists(benign_path):
        print(f"\n❌ 错误: 良性数据目录不存在: {benign_path}")
        return
    
    if not os.path.exists(malignant_path):
        print(f"\n❌ 错误: 恶性数据目录不存在: {malignant_path}")
        return
    
    print(f"\n✅ 数据目录检查通过")

    # ========================================================================
    # 第1步：数据集划分（训练集、验证集、测试集）
    # ========================================================================
    print_section("第1步：数据集划分")
    try:        
        # 分割数据集
        split_result = split_dataset(
            benign_path=benign_path,
            malignant_path=malignant_path,
            output_dir=SPLIT_DATA_DIR,
            split_ratio=0.9,
            random_seed=42
        )
        
        print("\n✅ 数据集划分完成！")
        print(f"  训练+验证集: {split_result['train_val_count']} 张")
        print(f"    - 良性: {split_result['train_val_benign']} 张")
        print(f"    - 恶性: {split_result['train_val_malignant']} 张")
        print(f"  测试集: {split_result['test_count']} 张")
        print(f"    - 良性: {split_result['test_benign']} 张")
        print(f"    - 恶性: {split_result['test_malignant']} 张")
        print(f"  总计: {split_result['total_count']} 张")
        print(f"  输出目录: {split_result['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ 数据集划分失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    
    # ========================================================================
    # 第2步：准备交叉验证数据
    # ========================================================================
    print_section("第2步：准备交叉验证数据")      
    print(f"开始准备 {NUM_FOLDS} 折交叉验证数据...")
    
    try:
        cv_result = prepare_cross_validation(
            data_path=TRAIN_VAL_DIR,
            output_base=CV_DATA_DIR,
            n_splits=NUM_FOLDS,
            random_state=42
        )
        
        print("\n✅ 交叉验证数据准备完成！")
        print(f"  折数: {cv_result['n_splits']}")
        print(f"  总样本数: {cv_result['total_samples']}")
        print(f"    - 良性: {cv_result['benign_count']} 张")
        print(f"    - 恶性: {cv_result['malignant_count']} 张")
        print(f"  输出目录: {cv_result['output_base']}")
        
        # 打印每折的详细信息
        print(f"\n  各折详细信息:")
        for fold_info in cv_result['fold_details']:
            print(f"    Fold {fold_info['fold']}:")
            print(f"      训练集: {fold_info['train_count']} 张 "
                  f"(良性: {fold_info['train_benign']}, 恶性: {fold_info['train_malignant']})")
            print(f"      验证集: {fold_info['val_count']} 张 "
                  f"(良性: {fold_info['val_benign']}, 恶性: {fold_info['val_malignant']})")
        
    except Exception as e:
        print(f"\n❌ 交叉验证数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    
    # ========================================================================
    # 第3步：训练模型
    # ========================================================================
    print_section("第3步：训练模型")
    
    print(f"开始训练 {len(MODELS_TO_TRAIN)} 个模型，每个模型训练 {NUM_FOLDS} 折")
    print(f"总训练次数: {len(MODELS_TO_TRAIN)} × {NUM_FOLDS} = {len(MODELS_TO_TRAIN) * NUM_FOLDS}")
    
    try:
        train_result = train_all_models(
            model_list=MODELS_TO_TRAIN,
            cv_data_dir=CV_DATA_DIR,
            output_dir=MODELS_DIR,            
            num_folds=NUM_FOLDS,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            initial_lr=INITIAL_LR,
            device=DEVICE,
            num_classes=NUM_CLASSES,
            num_workers=NUM_WORKERS,
            patience=PATIENCE
        )
        
        print("\n✅ 所有模型训练完成！")
        print(f"  总训练时间: {train_result['total_time']:.2f} 分钟")
        print(f"  训练记录数: {len(train_result['training_logs'])}")
        print(f"  模型保存目录: {train_result['output_dir']}")
    except Exception as e:
        print(f"\n❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # ========================================================================
    # 第4步：评估模型
    # ========================================================================
    print_section("第4步：评估模型")
    
    print(f"开始评估所有训练好的模型...")
    print(f"  交叉验证数据: {CV_DATA_DIR}")
    print(f"  测试集数据: {TEST_DIR}")
    print(f"  模型权重: {MODELS_DIR}")
    
    try:
        eval_result = evaluate_all_models(
            model_list=MODELS_TO_TRAIN,  # 评估刚才训练的模型
            cv_data_dir=CV_DATA_DIR,
            test_dir=TEST_DIR,
            models_dir=MODELS_DIR,
            output_dir=RESULTS_DIR,
            num_folds=NUM_FOLDS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            save_summary=True
        )
        
        if eval_result:
            print("\n✅ 所有模型评估完成！")
            print(f"  总评估时间: {eval_result['total_time']:.2f} 分钟")
            print(f"  评估模型数: {eval_result['num_models']}")
            print(f"  结果保存目录: {eval_result['output_dir']}")
            
            # 显示关键结果文件
            print("\n📊 生成的关键结果文件:")
            print(f"  1. 所有模型汇总: {os.path.join(RESULTS_DIR, 'all_models_summary.csv')}")
            print(f"  2. 性能热图: {os.path.join(RESULTS_DIR, 'all_models_heatmap.png')}")
            print(f"  3. Top4模型汇总: {os.path.join(RESULTS_DIR, 'top4_models_summary.csv')}")
            print(f"  4. Top4模型热图: {os.path.join(RESULTS_DIR, 'top4_models_heatmap.png')}")
        else:
            print("\n⚠️ 评估未返回结果")
        
    except Exception as e:
        print(f"\n❌ 模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print_section("流程完成")


    # ========================================================================
    # 第5步：总结
    # ========================================================================
    print_section("流程完成总结")
    
    print("🎉 完整的训练评估流程已完成！\n")
    
    print("📁 输出目录结构:")
    print(f"{EXPERIMENT_DIR}/")
    print(f"  ├── split_data/                    # 划分后的数据集")
    print(f"  │   ├── train_val/                 # 训练+验证集")
    print(f"  │   │   ├── benign/")
    print(f"  │   │   └── malignant/")
    print(f"  │   ├── test/                      # 测试集")
    print(f"  │   │   ├── benign/")
    print(f"  │   │   └── malignant/")
    print(f"  │   └── split_summary.csv")
    print(f"  ├── cross_validation_data/         # 交叉验证数据")
    print(f"  │   ├── fold_1/")
    print(f"  │   │   ├── train/")
    print(f"  │   │   └── val/")
    print(f"  │   ├── fold_2/")
    print(f"  │   ├── .../")
    print(f"  │   └── cv_summary.csv")
    print(f"  ├── train_val_models/              # 训练好的模型")
    print(f"  │   ├── ResNet50/")
    print(f"  │   │   ├── resnet50_fold_1.pth")
    print(f"  │   │   ├── resnet50_fold_2.pth")
    print(f"  │   │   └── .../")
    print(f"  │   └── training_log.xlsx")
    print(f"  └── evaluation_results/            # 评估结果")
    print(f"      ├── ResNet50/")
    print(f"      │   ├── fold_1/")
    print(f"      │   │   ├── train_probabilities.csv")
    print(f"      │   │   ├── train_metrics.csv")
    print(f"      │   │   ├── train_confusion_matrix.png")
    print(f"      │   │   ├── val_probabilities.csv")
    print(f"      │   │   ├── val_metrics.csv")
    print(f"      │   │   ├── val_confusion_matrix.png")
    print(f"      │   │   ├── test_probabilities.csv")
    print(f"      │   │   ├── test_metrics.csv")
    print(f"      │   │   └── test_confusion_matrix.png")
    print(f"      │   ├── fold_2/")
    print(f"      │   ├── .../")
    print(f"      │   ├── roc_curves/")
    print(f"      │   │   ├── roc_curve_fold_1.png")
    print(f"      │   │   └── .../")
    print(f"      │   ├── ResNet50_detailed_results.csv")
    print(f"      │   └── ResNet50_cross_validation_results.csv")
    print(f"      ├── all_models_summary.csv")
    print(f"      ├── all_models_heatmap.png")
    print(f"      ├── top4_models_summary.csv")
    print(f"      └── top4_models_heatmap.png")
    
    print("\n⏱️  时间统计:")
    if 'split_result' in locals():
        print(f"  数据划分: 完成")
    if 'cv_result' in locals():
        print(f"  交叉验证准备: 完成")
    if 'train_result' in locals():
        print(f"  训练总时间: {train_result['total_time']:.2f} 分钟")
    if 'eval_result' in locals() and eval_result:
        print(f"  评估总时间: {eval_result['total_time']:.2f} 分钟")
    
    print("\n📊 下一步建议:")
    print("  1. 查看 all_models_summary.csv 了解所有模型的性能")
    print("  2. 查看 all_models_heatmap.png 可视化性能对比")
    print("  3. 查看 top4_models_summary.csv 了解最佳模型")
    print("  4. 进入各模型文件夹查看详细的fold级别结果")
    print("  5. 查看 ROC 曲线和混淆矩阵进行深入分析")
    print("  6. 查看 training_log.xlsx 了解训练过程详情")
    
    print("\n" + "="*80)
    print("  流程结束")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序异常退出: {e}")
        import traceback
        traceback.print_exc()