"""
快速测试流程（使用默认参数）
"""

import torch
from shilab_classifier import (
    split_dataset,
    prepare_cross_validation,
    train_all_models,
    evaluate_all_models
)


def main():
    # 配置路径（请修改为您的实际路径）
    RAW_DATA_DIR = r"H:\0UR_classification_model\raw_data"
    OUTPUT_DIR = r"H:\0UR_classification_model\quick_test"
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 只训练2个模型用于快速测试
    models = ['ResNet50', 'VGG16']
    
    print("="*60)
    print("快速测试流程")
    print("="*60)
    
    # 步骤1: 划分数据集
    print("\n[1/4] 划分数据集...")
    split_result = split_dataset(
        data_dir=RAW_DATA_DIR,
        output_dir=f"{OUTPUT_DIR}/split_data"
    )
    print(f"✅ 完成！总计 {split_result['total_count']} 张图像")
    
    # 步骤2: 准备交叉验证
    print("\n[2/4] 准备交叉验证数据...")
    cv_result = prepare_cross_validation(
        train_dir=f"{OUTPUT_DIR}/split_data/train",
        val_dir=f"{OUTPUT_DIR}/split_data/val",
        output_dir=f"{OUTPUT_DIR}/cv_data"
    )
    print(f"✅ 完成！准备了 {cv_result['num_folds']} 折数据")
    
    # 步骤3: 训练模型
    print("\n[3/4] 训练模型...")
    train_result = train_all_models(
        model_list=models,
        cv_data_dir=f"{OUTPUT_DIR}/cv_data",
        output_dir=f"{OUTPUT_DIR}/models",
        device=device
    )
    print(f"✅ 完成！用时 {train_result['total_time']:.2f} 分钟")
    
    # 步骤4: 评估模型
    print("\n[4/4] 评估模型...")
    eval_result = evaluate_all_models(
        model_list=models,
        cv_data_dir=f"{OUTPUT_DIR}/cv_data",
        test_dir=f"{OUTPUT_DIR}/split_data/test",
        models_dir=f"{OUTPUT_DIR}/models",
        output_dir=f"{OUTPUT_DIR}/results",
        device=device
    )
    print(f"✅ 完成！用时 {eval_result['total_time']:.2f} 分钟")
    
    print("\n" + "="*60)
    print("🎉 所有步骤完成！")
    print(f"📁 结果保存在: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
