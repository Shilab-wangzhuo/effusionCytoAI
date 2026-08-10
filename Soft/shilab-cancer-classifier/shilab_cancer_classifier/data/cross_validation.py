"""
5折交叉验证数据准备模块

🔴 Bug 修复（影响正确性）
#	位置	原代码	修改后
1	文件收集	image_files 无排序	image_files = sorted(image_files)
2	输出目录	os.makedirs(..., exist_ok=True) 不清空旧数据	clear_directory() 先清空再创建
3	文件复制	shutil.copy2() 直接覆盖同名文件	safe_copy() 检测冲突后再复制

🟡 功能增强（影响可用性）
#	位置	原代码	修改后
4	函数入口	无路径验证，glob 静默返回空列表	验证 data_path 存在且包含图像
5	折循环内	简单 print 文字	收集结构化数据到 fold_logs
6	函数末尾	无汇总表格	打印多级列名汇总表格
7	日志保存	单 Sheet Excel，列结构简单	两个 Sheet：折划分汇总 + 明细
8	返回值	无返回值（隐式 None）	返回含 df_summary 的字典
"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from glob import glob
import shutil
from datetime import datetime

from .dataset_splitter import clear_directory


# 支持的图像扩展名白名单
VALID_IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']


def prepare_cross_validation(data_path: str, output_base: str, n_splits: int = 5, random_state: int = 42):
    """
    准备N折交叉验证数据集

    Args:
        data_path (str):    原始数据路径（结构：data_path/类别名/图像文件）
        output_base (str):  输出基础路径
        n_splits (int):     折数，默认5
        random_state (int): 随机种子，默认42，保证完美复现

    Returns:
        dict: 包含交叉验证统计信息的字典

    Raises:
        ValueError:      数据路径不存在或未找到图像文件
        FileExistsError: 复制时发现同名文件冲突
    """
    print(f"开始准备 {n_splits} 折交叉验证数据...")
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_base}")

    # 验证输入路径
    if not os.path.exists(data_path):
        raise ValueError(f"数据路径不存在: {data_path}")

    # sorted() 固定文件列表顺序，保证复现性
    image_files = []
    for ext in VALID_IMAGE_EXTENSIONS:
        image_files.extend(glob(os.path.join(data_path, '**', ext), recursive=True))
    image_files = sorted(image_files)

    if not image_files:
        raise ValueError(f"在 {data_path} 中未找到任何图像文件！")
    print(f"找到 {len(image_files)} 张图像")

    # 提取类别信息（父目录名即类别名）
    file_paths = []
    classes    = []
    for img_path in image_files:
        class_name = os.path.basename(os.path.dirname(img_path))
        file_paths.append(img_path)
        classes.append(class_name)

    df = pd.DataFrame({'file_path': file_paths, 'class': classes})
    print(f"\n类别分布:")
    print(df['class'].value_counts())

    # 清空输出目录
    print(f"\n清空输出目录...")
    clear_directory(output_base)

    min_class_count = int(df['class'].value_counts().min())
    if min_class_count < n_splits:
        raise ValueError(
            f"n_splits={n_splits} is larger than the smallest class count "
            f"({min_class_count}). Reduce n_splits or add more samples."
        )

    kf             = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    unique_classes = sorted(df['class'].unique())
    fold_logs      = []
    split_time     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n开始划分 {n_splits} 折...")
    print("=" * 50)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df['class'])):
        fold_dir  = os.path.join(output_base, f'fold_{fold + 1}')
        train_dir = os.path.join(fold_dir, 'train')
        val_dir   = os.path.join(fold_dir, 'val')

        for cls in unique_classes:
            clear_directory(os.path.join(train_dir, cls))
            clear_directory(os.path.join(val_dir,   cls))

        train_data = df.iloc[train_idx]
        val_data   = df.iloc[val_idx]

        def safe_copy(src: str, dst: str):
            if os.path.exists(dst):
                raise FileExistsError(
                    f"\n❌ 文件名冲突！\n"
                    f"   目标路径: {dst}\n"
                    f"   源文件:   {src}"
                )
            shutil.copy2(src, dst)

        for _, row in train_data.iterrows():
            src = row['file_path']
            dst = os.path.join(train_dir, row['class'], os.path.basename(src))
            safe_copy(src, dst)

        for _, row in val_data.iterrows():
            src = row['file_path']
            dst = os.path.join(val_dir, row['class'], os.path.basename(src))
            safe_copy(src, dst)

        # 统计每个类别的数量
        train_dist = train_data['class'].value_counts().to_dict()
        val_dist   = val_data['class'].value_counts().to_dict()

        # ✅ 按照表格结构收集每折数据
        row_data = {'Fold': f'fold_{fold + 1}'}
        for cls in unique_classes:
            row_data[f'Training_{cls}']   = train_dist.get(cls, 0)
            row_data[f'Validation_{cls}'] = val_dist.get(cls, 0)
        row_data['Training_Total']   = len(train_data)
        row_data['Validation_Total'] = len(val_data)
        fold_logs.append(row_data)

        print(f"Fold {fold + 1}: 训练集={len(train_data)}, 验证集={len(val_data)}")

    # ✅ 生成结构化汇总表格并打印
    df_summary = pd.DataFrame(fold_logs).set_index('Fold')

    # 构建多级列名，与图片表格一致
    col_map = {}
    for cls in unique_classes:
        col_map[f'Training_{cls}']   = ('Training Set',   cls.capitalize())
        col_map[f'Validation_{cls}'] = ('Validation Set', cls.capitalize())
    col_map['Training_Total']   = ('Training Set',   'Total')
    col_map['Validation_Total'] = ('Validation Set', 'Total')

    df_display = df_summary.rename(columns=col_map)
    df_display.columns = pd.MultiIndex.from_tuples(df_display.columns)

    print("\n" + "=" * 50)
    print("各折数据划分汇总：")
    print("=" * 50)
    print(df_display.to_string())
    print("=" * 50)

    # ✅ 保存 Excel（多级列名 + 划分时间）
    log_path = os.path.join(output_base, "cross_validation_log.xlsx")
    with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
        df_display.to_excel(writer, sheet_name='折划分汇总')

        # 额外保存一张带划分时间的明细表
        df_detail = pd.DataFrame(fold_logs)
        df_detail.insert(1, '划分时间', split_time)
        df_detail.to_excel(writer, sheet_name='明细', index=False)

    print(f"\n{n_splits} 折交叉验证数据集准备完成！")
    print(f"输出目录: {output_base}")
    print(f"日志已保存: {log_path}")

    return {
        'n_splits':   n_splits,
        'total':      len(df),
        'log_path':   log_path,
        'df_summary': df_display,
    }
