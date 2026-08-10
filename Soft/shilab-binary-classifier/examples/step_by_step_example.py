"""
分步执行示例
每个步骤可以单独运行
"""

import torch
from shilab_classifier import (
    split_dataset,
    prepare_cross_validation,
    train_all_models,
    evaluate_all_models
)


# 配置（全局变量）
RAW_DATA_DIR = r"H:\0UR_classification_model\raw_data"
BASE_DIR = r"H:\0UR_classification_model\experiment"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def step1_split_dataset():
    """步骤1: 划分数据集"""
    print("\n" + "="*60)
    print("步骤1: 划分数据集")
    print("="*60)
    
    result = split_dataset(
        data_dir=RAW_DATA_DIR,
        output_dir=f"{BASE_DIR}/split_data",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print(f"\n✅ 数据集划分完成！")
    print(f"  训练集: {result['train_count']}")
    print(f"  验证集: {result['val_count']}")
    print(f"  测试集: {result['test_count']}")
    
    return result


def step2_prepare_cv():
    """步骤2: 准备交叉验证数据"""
    print("\n" + "="*60)
    print("步骤2: 准备交叉验证数据")
    print("="*60)
    
    result = prepare_cross_validation(
        train_dir=f"{BASE_DIR}/split_data/train",
        val_dir=f"{BASE_DIR}/split_data/val",
        output_dir=f"{BASE_DIR}/cv_data",
        num_folds=5,
        random_seed=42
    )
    
    print(f"\n✅ 交叉验证数据准备完成！")
    print(f"  折数: {result['num_folds']}")
    print(f"  总样本: {result['total_samples']}")
    
    return result


def step3_train_models():
    """步骤3: 训练模型"""
    print("\n" + "="*60)
    print("步骤3: 训练模型")
    print("="*60)
    
    # 选择要训练的模型
    models = ['ResNet50', 'VGG16']
    
    result = train_all_models(
        model_list=models,
        cv_data_dir=f"{BASE_DIR}/cv_data",
        output_dir=f"{BASE_DIR}/models",
        num_epochs=50,
        device=DEVICE
    )
    
    print(f"\n✅ 模型训练完成！")
    print(f"  用时: {result['total_time']:.2f} 分钟")
    
    return result


def step4_evaluate_models():
    """步骤4: 评估模型"""
    print("\n" + "="*60)
    print("步骤4: 评估模型")
    print("="*60)
    
    models = ['ResNet50', 'VGG16']
    
    result = evaluate_all_models(
        model_list=models,
        cv_data_dir=f"{BASE_DIR}/cv_data",
        test_dir=f"{BASE_DIR}/split_data/test",
        models_dir=f"{BASE_DIR}/models",
        output_dir=f"{BASE_DIR}/results",
        device=DEVICE
    )
    
    print(f"\n✅ 模型评估完成！")
    print(f"  用时: {result['total_time']:.2f} 分钟")
    print(f"  评估了 {result['num_models']} 个模型")
    
    return result


def main():
    """主函数：可以选择运行哪些步骤"""
    print("="*60)
    print("分步执行示例")
    print("="*60)
    print("\n请选择要执行的步骤:")
    print("  1. 划分数据集")
    print("  2. 准备交叉验证数据")
    print("  3. 训练模型")
    print("  4. 评估模型")
    print("  5. 执行所有步骤")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-5): ")
    
    if choice == '1':
        step1_split_dataset()
    elif choice == '2':
        step2_prepare_cv()
    elif choice == '3':
        step3_train_models()
    elif choice == '4':
        step4_evaluate_models()
    elif choice == '5':
        step1_split_dataset()
        step2_prepare_cv()
        step3_train_models()
        step4_evaluate_models()
        print("\n🎉 所有步骤完成！")
    elif choice == '0':
        print("退出程序")
    else:
        print("无效选项")


if __name__ == "__main__":
    main()
