"""
模型训练示例

演示如何使用shilab_classifier训练19个SOTA模型
"""

import torch
from shilab_classifier import train_all_models, SUPPORTED_MODELS

if __name__ == "__main__":
    print("="*60)
    print("SOTA模型训练程序")
    print("="*60)
    
    # 配置参数
    cv_data_dir = r"path/to/cross_validation_data"  # 交叉验证数据目录
    output_dir = r"path/to/trained_models"  # 模型输出目录
    
    # 选择要训练的模型（可以选择部分或全部）
    model_list = [
        'ResNet50',
        'VGG16',
        'DenseNet121',
        'EfficientNet_B0'
    ]
    # 或者训练所有支持的模型
    # model_list = SUPPORTED_MODELS
    
    # 训练参数
    num_folds = 5
    num_epochs = 50
    batch_size = 32
    initial_lr = 0.001
    patience = 10
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n✅ 交叉验证数据路径: {cv_data_dir}")
    print(f"✅ 模型输出路径: {output_dir}")
    print(f"✅ 训练设备: {device}")
    print(f"✅ 模型列表: {model_list}")
    print(f"✅ 折数: {num_folds}")
    print(f"✅ 训练轮数: {num_epochs}")
    
    # 开始训练
    # 使用默认: num_folds=5, num_epochs=50, batch_size=16, etc.
    result = train_all_models(
        model_list=model_list,
        cv_data_dir=cv_data_dir,
        output_dir=output_dir,
        device=device  # 只需要指定设备
    )

    result = train_all_models(
        model_list=model_list,
        cv_data_dir=cv_data_dir,
        output_dir=output_dir,
        num_folds=num_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        initial_lr=initial_lr,
        device=device,
        num_classes=2,
        num_workers=num_workers,
        patience=patience,
        save_log=True
    )
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("="*60)
    print(f"\n📊 统计信息:")
    print(f"  总训练时间: {result['total_time']:.2f} 分钟")
    print(f"  输出目录: {result['output_dir']}")
    print(f"  训练记录数: {len(result['training_logs'])}")
