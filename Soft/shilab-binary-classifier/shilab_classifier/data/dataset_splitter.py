"""
数据集分割模块（修复版 v3）
提供数据集分割功能，支持分层抽样
# #	改动点	原代码	修改后代码
# 1	随机性控制	全局 random.seed()	局部 random.Random(seed) 实例
# 2	目录清空	❌ 无，仅 makedirs	✅ 新增 clear_directory()
# 3	文件名处理	.replace(".svs","") 篡改文件名	✅ 直接保留原始文件名
# 4	冲突检测	❌ 无，静默覆盖	✅ os.path.exists() 检测后报错
# 5	文件过滤	❌ 无，处理所有文件	✅ 白名单过滤非图像文件
# 6	统计日志	第二次循环重新计算	✅ 同一循环内记录，与实际一致
"""

import os
import shutil
import random
from pathlib import Path
import pandas as pd
from datetime import datetime

# 图像文件扩展名白名单
VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
# ✅ 移除 '.svs'：您的源文件全是png，svs是病理切片原始格式，不应出现在源目录


def clear_directory(directory: str):
    """
    清空目录中的所有内容（保留目录本身）
    若目录不存在则创建
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"  已清空目录: {directory}")
    else:
        os.makedirs(directory)
        print(f"  已创建目录: {directory}")


def split_dataset(benign_path, malignant_path, output_dir,
                  split_ratio=0.9, random_seed=42):
    """
    将数据集分成训练验证集(train_val)和测试集(test)，使用分层抽样

    规则：
        - 良性图像：100% 进入训练验证集，测试集为空
        - 恶性图像：按 split_ratio 划分训练验证集和测试集

    参数:
        benign_path (str):    良性图像的路径
        malignant_path (str): 恶性图像的路径
        output_dir (str):     输出目录
        split_ratio (float):  恶性图像训练验证集的比例，默认为 0.9
        random_seed (int):    随机种子，默认为 42（保证完美复现）

    返回:
        dict: 包含分割统计信息的字典
    """
    # 使用局部 Random 实例，完全隔离全局随机状态，保证复现性
    rng = random.Random(random_seed)

    # 验证输入路径
    if not os.path.exists(benign_path):
        raise ValueError(f"良性图像路径不存在: {benign_path}")
    if not os.path.exists(malignant_path):
        raise ValueError(f"恶性图像路径不存在: {malignant_path}")

    split_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    split_logs = []

    # 定义目录结构
    train_val_benign_dir    = os.path.join(output_dir, "train_val", "benign")
    train_val_malignant_dir = os.path.join(output_dir, "train_val", "malignant")
    test_benign_dir         = os.path.join(output_dir, "test",      "benign")
    test_malignant_dir      = os.path.join(output_dir, "test",      "malignant")

    # 划分前清空所有目标目录，杜绝历史残留
    print("=" * 50)
    print("Step 1: 清空目标目录...")
    for directory in [train_val_benign_dir, train_val_malignant_dir,
                      test_benign_dir, test_malignant_dir]:
        clear_directory(directory)

    # 处理良性图像（split_ratio=1.0，全部进训练验证集）
    print("\n" + "=" * 50)
    print("Step 2: 处理良性图像（全部放入训练验证集）...")
    benign_log = _process_images(
        benign_path, train_val_benign_dir, test_benign_dir,
        split_ratio=1.0, rng=rng
    )
    split_logs.append({
        '划分时间':       split_time,
        '类别':          '良性',
        '输入路径':       benign_path,
        '输出路径':       output_dir,
        '分割比例':       '100% → 训练验证集',
        '训练验证集数量': benign_log['train_val'],
        '测试集数量':     benign_log['test'],
        '总数':          benign_log['total']
    })

    # 处理恶性图像（按 split_ratio 划分）
    print("\n" + "=" * 50)
    print(f"Step 3: 处理恶性图像（训练验证集比例={split_ratio*100:.1f}%）...")
    malignant_log = _process_images(
        malignant_path, train_val_malignant_dir, test_malignant_dir,
        split_ratio=split_ratio, rng=rng
    )
    split_logs.append({
        '划分时间':       split_time,
        '类别':          '恶性',
        '输入路径':       malignant_path,
        '输出路径':       output_dir,
        '分割比例':       f'{split_ratio*100:.1f}% → 训练验证集',
        '训练验证集数量': malignant_log['train_val'],
        '测试集数量':     malignant_log['test'],
        '总数':          malignant_log['total']
    })

    # 保存日志
    print("\n" + "=" * 50)
    train_val_dir = os.path.join(output_dir, "train_val")
    test_dir      = os.path.join(output_dir, "test")
    print(f"数据集分割完成！")
    print(f"  训练验证集保存在: {train_val_dir}")
    print(f"  测试集保存在:     {test_dir}")

    df = pd.DataFrame(split_logs)
    excel_path = os.path.join(output_dir, "dataset_split_log.xlsx")
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"  日志已保存到:     {excel_path}")

    return {
        'benign':    benign_log,
        'malignant': malignant_log,
        'log_path':  excel_path
    }


def _process_images(source_dir, train_val_dir, test_dir, split_ratio, rng):
    """
    处理图像：分层抽样分割并复制到目标目录

    参数:
        source_dir (str):    源目录
        train_val_dir (str): 训练验证集目标目录
        test_dir (str):      测试集目标目录
        split_ratio (float): 训练验证集比例
        rng (random.Random): 局部随机实例，保证复现性
    """
    # sorted() 保证文件列表顺序与操作系统无关，是复现性的关键
    image_files = sorted([
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"  ⚠️  警告：{source_dir} 中未找到任何图像文件！")
        return {'train_val': 0, 'test': 0, 'total': 0}

    # 按扩展名分组（分层抽样）
    file_groups: dict[str, list[str]] = {}
    for file in image_files:
        ext = os.path.splitext(file)[1].lower()
        file_groups.setdefault(ext, []).append(file)

    train_val_files = []
    test_files      = []
    detail_logs     = []

    for ext, files in sorted(file_groups.items()):
        rng.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        tv = files[:split_idx]
        te = files[split_idx:]
        train_val_files.extend(tv)
        test_files.extend(te)
        detail_logs.append(
            f"  扩展名 {ext:6s}: 总数={len(files):4d}, "
            f"训练验证集={len(tv):4d}, 测试集={len(te):4d}"
        )

    print(f"  分层抽样结果: 训练验证集={len(train_val_files)}, "
          f"测试集={len(test_files)}, 合计={len(image_files)}")
    print("  分层详情:")
    for log in detail_logs:
        print(log)

    # ✅ 修复核心：直接保留原始文件名，不做任何替换
    def safe_copy(file: str, dst_dir: str):
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(dst_dir, file)      # ✅ 直接使用原始文件名

        if os.path.exists(dst_path):
            raise FileExistsError(
                f"\n❌ 文件名冲突（源目录中存在重名文件）！\n"
                f"   目标路径:   {dst_path}\n"
                f"   当前源文件: {src_path}\n"
            )
        shutil.copy2(src_path, dst_path)

    for file in train_val_files:
        safe_copy(file, train_val_dir)
    for file in test_files:
        safe_copy(file, test_dir)

    print(f"  ✅ 文件复制完成")

    return {
        'train_val': len(train_val_files),
        'test':      len(test_files),
        'total':     len(image_files)
    }
