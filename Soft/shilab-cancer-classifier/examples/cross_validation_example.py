"""
5折交叉验证示例

演示如何使用shilab_cancer_classifier进行5折交叉验证数据准备
"""

from shilab_cancer_classifier import prepare_cross_validation

if __name__ == "__main__":
    print("="*60)
    print("5折交叉验证数据准备程序")
    print("="*60)
    
    # 设置路径
    data_path = r"path/to/train_val"  # 包含benign和malignant子文件夹的路径
    output_base = r"path/to/cross_validation_output"
    
    # 配置参数
    n_splits = 5
    random_state = 42
    
    print(f"\n✅ 原始数据路径: {data_path}")
    print(f"✅ 输出路径: {output_base}")
    print(f"✅ 折数: {n_splits}")
    print(f"✅ 随机种子: {random_state}")
    
    # 生成交叉验证数据
    result = prepare_cross_validation(
        data_path=data_path,
        output_base=output_base,
        n_splits=n_splits,
        random_state=random_state
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("✅ 数据准备完成！")
    print("="*60)
    print(f"\n📊 统计信息:")
    print(f"  总样本数: {result['total_samples']}")
    print(f"  良性样本数: {result['benign_samples']}")
    print(f"  恶性样本数: {result['malignant_samples']}")
    print(f"  折数: {result['n_splits']}")
    print(f"  输出路径: {result['output_base']}")
    print(f"  日志文件: {result['log_path']}")
