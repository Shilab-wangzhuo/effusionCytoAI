"""
模型评估示例

演示如何使用shilab_classifier评估训练好的模型
"""

import torch
from shilab_classifier import evaluate_all_models

if __name__ == "__main__":
    print("="*60)
    print("模型评估程序")
    print("="*60)
    
    # 配置路径
    cv_data_dir = r"path/to/cross_validation_data"  # 交叉验证数据目录
    test_dir = r"path/to/test_data"  # 测试集目录
    models_dir = r"path/to/trained_models"  # 训练好的模型目录
    output_dir = r"path/to/evaluation_results"  # 评估结果输出目录
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n✅ 交叉验证数据路径: {cv_data_dir}")
    print(f"✅ 测试集路径: {test_dir}")
    print(f"✅ 模型路径: {models_dir}")
    print(f"✅ 输出路径: {output_dir}")
    print(f"✅ 评估设备: {device}")
    
    # 方案1: 评估所有模型（自动检测models_dir中的所有模型）
    result = evaluate_all_models(
        model_list=None,  # None表示自动检测所有模型
        cv_data_dir=cv_data_dir,
        test_dir=test_dir,
        models_dir=models_dir,
        output_dir=output_dir,
        device=device
    )
    
    # 方案2: 只评估指定的模型
    # result = evaluate_all_models(
    #     model_list=['ResNet50', 'VGG16', 'DenseNet121'],
    #     cv_data_dir=cv_data_dir,
    #     test_dir=test_dir,
    #     models_dir=models_dir,
    #     output_dir=output_dir,
    #     device=device
    # )
    
    print("\n" + "="*60)
    print("✅ 评估完成！")
    print("="*60)
    if result:
        print(f"\n📊 统计信息:")
        print(f"  总评估时间: {result['total_time']:.2f} 分钟")
        print(f"  评估模型数: {result['num_models']}")
        print(f"  输出目录: {result['output_dir']}")
