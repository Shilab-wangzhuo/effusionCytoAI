
"""
数据集分割示例

演示如何使用shilab_classifier进行数据集分割
"""

from shilab_classifier import split_dataset

if __name__ == "__main__":
    # 设置路径
    benign_path = r"path/to/benign/images"
    malignant_path = r"path/to/malignant/images"
    output_dir = r"path/to/output"
    
    # 分割数据集
    result = split_dataset(
        benign_path=benign_path,
        malignant_path=malignant_path,
        output_dir=output_dir,
        split_ratio=0.9,
        random_seed=42
    )
    
    # 打印结果
    print("\n=== 分割结果 ===")
    print(f"良性图像统计: {result['benign']}")
    print(f"恶性图像统计: {result['malignant']}")
    print(f"日志文件: {result['log_path']}")
