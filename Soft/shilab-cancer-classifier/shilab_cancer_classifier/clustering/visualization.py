# visualization/visualization.py
# 用来存放可视化相关的函数

# 0. 导入所有依赖库
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from tqdm import tqdm
import umap  # 确保你安装了 umap-learn
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# 1. 动态类别配置生成函数
# ==============================================================================

# 预设调色板（供癌种颜色循环使用，最多支持 10 种癌种）
_CANCER_COLOR_PALETTE = [
    ( 31, 119, 180),   # 蓝
    (255, 127,  14),   # 橙
    ( 44, 160,  44),   # 绿
    (214,  39,  40),   # 红
    (148, 103, 189),   # 紫
    (140,  86,  75),   # 棕
    (227, 119, 194),   # 粉
    (127, 127, 127),   # 灰
    (188, 189,  34),   # 黄绿
    ( 23, 190, 207),   # 青
]

# marker 选择差异大的形状，避免多个三角形难以区分
_CANCER_MARKER_PALETTE = [
    'o',   # 圆形
    's',   # 方形
    'D',   # 菱形
    '*',   # 星形
    'P',   # 加号填充
    'X',   # X填充
    '^',   # 上三角
    'h',   # 六边形
    '8',   # 八边形
    'v',   # 下三角
]


def build_category_config(cancer_dirs: dict, include_candidate: bool = False):
    """
    根据 cancer_dirs 动态生成类别名称、颜色和 marker。

    参数:
        cancer_dirs (dict):
            有序字典，键为癌种名称，值为路径。
            例如 {'Gastrointestinal_Breast': '...', 'Lung': '...'}

        include_candidate (bool):
            False（默认）:
                用于 reference-only 多癌种数据，source_label 直接从 0 开始对应癌种。
                映射为:
                    0 -> 第1个癌种
                    1 -> 第2个癌种
                    ...
            True:
                用于包含 Candidate 的混合场景（reference + candidate 混合可视化）。
                映射为:
                    0 -> Candidate（固定深蓝色，方形）
                    1 -> 第1个癌种
                    2 -> 第2个癌种
                    ...

    返回:
        category_names   (dict): {0: '癌种名', 1: '癌种名', ...}
        category_colors  (dict): {0: (R,G,B), 1: (R,G,B), ...}
        category_markers (dict): {0: 'marker字符', ...}
    """
    cancer_names = list(cancer_dirs.keys())

    category_names   = {}
    category_colors  = {}
    category_markers = {}

    if include_candidate:
        # 0 固定为 Candidate
        category_names[0]   = 'Candidate'
        category_colors[0]  = (40, 120, 181)   # 固定深蓝色
        category_markers[0] = 's'             # 固定方形
        offset = 1
    else:
        # reference-only：0 直接对应第一个癌种
        offset = 0

    for i, name in enumerate(cancer_names):
        idx = i + offset
        category_names[idx]   = name
        category_colors[idx]  = _CANCER_COLOR_PALETTE[i % len(_CANCER_COLOR_PALETTE)]
        category_markers[idx] = _CANCER_MARKER_PALETTE[i % len(_CANCER_MARKER_PALETTE)]

    return category_names, category_colors, category_markers


def set_plot_font():
    """设置绘图字体参数"""
    plt.rcParams['font.family']      = 'sans-serif'
    plt.rcParams['font.sans-serif']  = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight']      = 'bold'


# ==============================================================================
# 2. 工具函数
# ==============================================================================

def count_non_background_pixels(image_path):
    """
    计算图像中非(225,225,225)背景像素的数量，作为细胞大小的估计。
    """
    try:
        img       = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        bg_mask   = np.all(img_array == [225, 225, 225], axis=2)
        return int(np.sum(~bg_mask))
    except Exception as e:
        print(f"计算图像 {image_path} 的非背景像素数量时出错: {e}")
        return 0


def _color_map_hex(category_colors: dict) -> dict:
    """将 RGB tuple 颜色字典转换为 hex 字符串字典"""
    return {i: f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for i, c in category_colors.items()}


def _to_int_list(arr):
    """将 source_labels / labels 统一转换为 Python int 列表，避免 np.int64 类型匹配问题"""
    return [int(x) for x in arr]


# ==============================================================================
# 2-1. 统计每个簇中各类别的数量（堆叠柱状图）
# ==============================================================================

def analyze_cluster_composition(labels, source_labels, image_paths,
                                method_name, save_dir,
                                category_names, category_colors):
    """
    统计每个簇中各类别的数量，并绘制堆叠柱状图。
    同时保存 .tif 和 .png 两种格式，方便预览。

    参数:
        labels          : 聚类标签（list/array）
        source_labels   : 源标签（0, 1, 2, ...）
        image_paths     : 图像路径列表
        method_name     : 聚类方法名称（用于标题和文件名）
        save_dir        : 保存目录（直接保存在此目录下，不在子目录）
        category_names  : 由 build_category_config 生成的类别名称字典
        category_colors : 由 build_category_config 生成的类别颜色字典

    返回:
        cluster_composition: 每个簇的类别分布字典
    """
    # 统一类型，避免 np.int64 与 int 混用导致匹配失败
    labels        = _to_int_list(labels)
    source_labels = _to_int_list(source_labels)

    os.makedirs(save_dir, exist_ok=True)

    color_map       = _color_map_hex(category_colors)
    unique_clusters = sorted(set(labels))

    results_df = pd.DataFrame({
        'image_path'  : image_paths,
        'source_label': source_labels,
        'cluster'     : labels
    })
    results_df['category'] = results_df['source_label'].map(category_names)

    existing_categories = set(int(x) for x in results_df['source_label'].unique())

    # 统计每个簇的类别分布
    cluster_composition = {}
    for cluster in unique_clusters:
        cluster_samples = results_df[results_df['cluster'] == cluster]
        category_counts = cluster_samples['source_label'].value_counts().to_dict()
        cluster_composition[cluster] = {
            'total': len(cluster_samples),
            'categories': {
                category_names[i]: category_counts.get(i, 0)
                for i in sorted(category_names.keys())
                if i in existing_categories
            }
        }

    # 生成汇总 DataFrame
    summary_data = []
    for cluster, data in cluster_composition.items():
        row = {'Cluster': cluster, 'Total': data['total']}
        row.update(data['categories'])
        summary_data.append(row)
    summary_df = pd.DataFrame(summary_data)

    # 保存 CSV
    csv_path = os.path.join(save_dir, f"{method_name}_cluster_composition.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"聚类组成摘要 CSV 已保存到: {csv_path}")

    set_plot_font()

    fig = plt.figure(figsize=(18, 12))
    ax  = fig.add_axes([0.1, 0.1, 0.65, 0.8])

    clusters        = summary_df['Cluster'].tolist()
    bottom          = np.zeros(len(clusters))
    legend_elements = []

    for i in sorted(category_names.keys()):
        category = category_names[i]
        if i in existing_categories and category in summary_df.columns:
            values = summary_df[category].values.astype(float)
            ax.bar(clusters, values, bottom=bottom, color=color_map[i])
            bottom += values
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=color_map[i], label=category)
            )

    ax.set_xlabel('Cluster', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('Number of Samples', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_title(f'{method_name} - Cluster Composition', fontsize=32, fontweight='bold', pad=20)
    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters, rotation=90 if len(clusters) > 10 else 0)
    ax.set_xlim(-0.6, len(clusters) - 0.4)
    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    legend_ax = fig.add_axes([0.76, 0.5, 0.1, 0.4])
    legend_ax.axis('off')
    legend = legend_ax.legend(
        handles=legend_elements,
        loc='center left',
        title="Categories",
        title_fontsize=18,
        prop={'weight': 'bold', 'size': 18},
        labelspacing=0.8,
        borderpad=1.0,
        handletextpad=0.5,
        frameon=True
    )
    legend_ax.add_artist(legend)

    # 同时保存 tif 和 png，方便预览确认
    tif_path = os.path.join(save_dir, f"{method_name}_cluster_composition.tif")
    # png_path = os.path.join(save_dir, f"{method_name}_cluster_composition.png")
    plt.savefig(tif_path, dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
    # plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"聚类组成柱状图已保存到: {tif_path}")
    # print(f"聚类组成柱状图 PNG 副本已保存到: {png_path}")

    return cluster_composition


# ==============================================================================
# 2-2. 全局 UMAP 可视化 —— 颜色区分簇，形状区分类别
# ==============================================================================

def visualize_all_clusters_without_images(features, labels, source_labels,
                                          method_name, save_dir,
                                          category_names, category_colors, category_markers,
                                          trained_umap=None):
    """
    全局 UMAP 可视化：颜色区分簇，点形状区分类别。

    参数:
        features        : 特征矩阵 (N, D)
        labels          : 聚类标签 (N,)
        source_labels   : 源标签 (N,)，值为 category_names 的键
        method_name     : 方法名称（用于标题和文件名）
        save_dir        : 保存目录（图像保存在 save_dir/cluster_images/ 下）
        category_names  : 由 build_category_config 生成
        category_colors : 由 build_category_config 生成
        category_markers: 由 build_category_config 生成
        trained_umap    : 可选，预训练 UMAP 模型

    返回:
        reducer: 训练好的 UMAP 模型
    """
    print(f"为 {method_name} 创建全局 UMAP 可视化（颜色=簇，形状=类别）...")

    # 统一类型
    labels        = _to_int_list(labels)
    source_labels = _to_int_list(source_labels)

    if trained_umap is None:
        reducer   = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        reducer   = trained_umap
        embedding = trained_umap.transform(features)

    unique_clusters = sorted(set(labels))
    cluster_colors  = cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    existing_cats   = set(source_labels)

    set_plot_font()
    fig = plt.figure(figsize=(18, 12))
    ax  = fig.add_axes([0.1, 0.1, 0.65, 0.8])

    for cluster_idx, cluster in enumerate(unique_clusters):
        c_indices = [i for i, l in enumerate(labels) if l == cluster]
        c_emb     = embedding[c_indices]
        c_sl      = [source_labels[i] for i in c_indices]

        for cat in existing_cats:
            cat_ii = [i for i, sl in enumerate(c_sl) if sl == cat]
            if cat_ii:
                ax.scatter(
                    c_emb[cat_ii, 0], c_emb[cat_ii, 1],
                    c=[cluster_colors[cluster_idx]],
                    marker=category_markers.get(cat, 'o'),
                    alpha=0.7, s=50,
                    edgecolors='black', linewidths=0.5
                )

    # 簇颜色图例
    cluster_legend = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cluster_colors[i],
                   markersize=16, label=f'Cluster {c}')
        for i, c in enumerate(unique_clusters)
    ]
    # 类别形状图例
    cat_legend = [
        plt.Line2D([0], [0], marker=category_markers[cat], color='black',
                   markersize=16, label=category_names[cat])
        for cat in sorted(category_markers.keys())
        if cat in existing_cats
    ]

    leg_ax1 = fig.add_axes([0.76, 0.5, 0.22, 0.4])
    leg_ax1.axis('off')
    leg_ax1.add_artist(leg_ax1.legend(
        handles=cluster_legend, title="Clusters",
        title_fontsize=18, prop={'weight': 'bold', 'size': 16},
        labelspacing=0.8, borderpad=1.0, handletextpad=0.5,
        frameon=True, loc='center left'
    ))

    leg_ax2 = fig.add_axes([0.76, 0.05, 0.22, 0.4])
    leg_ax2.axis('off')
    leg_ax2.add_artist(leg_ax2.legend(
        handles=cat_legend, title="Categories",
        title_fontsize=18, prop={'weight': 'bold', 'size': 16},
        labelspacing=0.8, borderpad=1.0, handletextpad=0.5,
        frameon=True, loc='center left'
    ))

    ax.set_title(f"{method_name} - All Clusters Distribution",
                 fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    out_dir = os.path.join(save_dir, 'cluster_images')
    os.makedirs(out_dir, exist_ok=True)
    tif_path = os.path.join(out_dir, f"{method_name}_all_clusters_distribution.tif")
    # png_path = os.path.join(out_dir, f"{method_name}_all_clusters_distribution.png")
    plt.savefig(tif_path, dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
    # plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {tif_path}")
    return reducer


# ==============================================================================
# 2-2-1. 全局 UMAP 可视化 —— 颜色区分类别，点全是圆点
# ==============================================================================

def visualize_all_clusters_without_images_1(features, labels, source_labels,
                                            method_name, save_dir,
                                            category_names, category_colors,
                                            trained_umap=None):
    """
    全局 UMAP 可视化：颜色区分类别，所有点使用圆形。

    参数:
        features        : 特征矩阵 (N, D)
        labels          : 聚类标签 (N,)
        source_labels   : 源标签 (N,)
        method_name     : 方法名称
        save_dir        : 保存目录（图像保存在 save_dir/cluster_images/ 下）
        category_names  : 由 build_category_config 生成
        category_colors : 由 build_category_config 生成
        trained_umap    : 可选，预训练 UMAP 模型

    返回:
        reducer: 训练好的 UMAP 模型
    """
    print(f"为 {method_name} 创建全局 UMAP 可视化（颜色=类别，形状=圆点）...")

    # 统一类型
    labels        = _to_int_list(labels)
    source_labels = _to_int_list(source_labels)

    if trained_umap is None:
        reducer   = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        reducer   = trained_umap
        embedding = trained_umap.transform(features)

    unique_clusters = sorted(set(labels))
    color_map       = _color_map_hex(category_colors)
    existing_cats   = set(source_labels)

    set_plot_font()
    fig = plt.figure(figsize=(18, 12))
    ax  = fig.add_axes([0.1, 0.1, 0.65, 0.8])

    for cluster in unique_clusters:
        c_indices = [i for i, l in enumerate(labels) if l == cluster]
        c_emb     = embedding[c_indices]
        c_sl      = [source_labels[i] for i in c_indices]

        for cat in existing_cats:
            cat_ii = [i for i, sl in enumerate(c_sl) if sl == cat]
            if cat_ii:
                ax.scatter(
                    c_emb[cat_ii, 0], c_emb[cat_ii, 1],
                    c=color_map[cat],
                    marker='o',
                    alpha=0.7, s=50,
                    edgecolors='black', linewidths=0.5
                )

    cat_legend = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_map[cat],
                   markersize=16, label=category_names[cat])
        for cat in sorted(category_colors.keys())
        if cat in existing_cats
    ]

    leg_ax = fig.add_axes([0.76, 0.3, 0.22, 0.4])
    leg_ax.axis('off')
    leg_ax.add_artist(leg_ax.legend(
        handles=cat_legend, loc='center left',
        title="Categories", title_fontsize=18,
        prop={'weight': 'bold', 'size': 16},
        labelspacing=0.8, borderpad=1.0, handletextpad=0.5,
        frameon=True
    ))

    ax.set_title(f"{method_name} - All Clusters Distribution",
                 fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    out_dir = os.path.join(save_dir, 'cluster_images')
    os.makedirs(out_dir, exist_ok=True)
    tif_path = os.path.join(out_dir, f"{method_name}_all_clusters_distribution_1.tif")
    # png_path = os.path.join(out_dir, f"{method_name}_all_clusters_distribution_1.png")
    plt.savefig(tif_path, dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
    # plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {tif_path}")
    return reducer


# ==============================================================================
# 2-3. 每个簇的 UMAP 图（缩略图大小反映细胞实际大小）
# ==============================================================================

def improved_visualize_clusters_with_images(features, labels, source_labels, image_paths,
                                            method_name, save_dir,
                                            category_names, category_colors,
                                            log_sizes=None,
                                            thumbnail_size=30,
                                            max_images=1000):
    """
    为每个簇创建 UMAP 可视化，并在点上显示图像缩略图。
    缩略图大小反映细胞的实际大小（对数变换后归一化）。

    参数:
        features        : 特征矩阵 (N, D)
        labels          : 聚类标签 (N,)
        source_labels   : 源标签 (N,)
        image_paths     : 图像路径列表 (N,)
        method_name     : 方法名称
        save_dir        : 保存目录（图像保存在 save_dir/cluster_images/ 下）
        category_names  : 由 build_category_config 生成
        category_colors : 由 build_category_config 生成
        log_sizes       : 可选，预计算的对数细胞大小 (N,)
        thumbnail_size  : 基础缩略图大小（像素）
        max_images      : 每个簇最大显示图像数
    """
    print(f"为 {method_name} 的每个簇创建缩略图可视化...")

    # 统一类型
    labels        = _to_int_list(labels)
    source_labels = _to_int_list(source_labels)

    color_map = _color_map_hex(category_colors)

    # 计算或使用传入的细胞大小
    if log_sizes is None:
        print("计算细胞大小...")
        cell_sizes = [count_non_background_pixels(p) for p in tqdm(image_paths, desc="计算细胞大小")]
        log_sizes  = np.log1p(np.array(cell_sizes))
    else:
        log_sizes = np.asarray(log_sizes, dtype=float)

    if np.max(log_sizes) > np.min(log_sizes):
        normalized         = (log_sizes - np.min(log_sizes)) / (np.max(log_sizes) - np.min(log_sizes))
        size_scale_factors = 0.5 + 1.5 * normalized
    else:
        size_scale_factors = np.ones_like(log_sizes)

    unique_clusters = sorted(set(labels))
    out_dir         = os.path.join(save_dir, 'cluster_images')
    os.makedirs(out_dir, exist_ok=True)

    for cluster in unique_clusters:
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster]

        if len(cluster_indices) > max_images:
            np.random.seed(42)
            selected_indices = list(np.random.choice(cluster_indices, max_images, replace=False))
        else:
            selected_indices = cluster_indices

        c_features     = features[selected_indices]
        c_source_labels = [source_labels[i] for i in selected_indices]
        c_image_paths  = [image_paths[i]    for i in selected_indices]
        c_size_factors = [size_scale_factors[i] for i in selected_indices]

        # 降维
        n = len(c_features)
        if n == 1:
            c_embedding = np.array([[0.0, 0.0]])
            print(f"  簇 {cluster}: 1 个样本，放在原点")
        elif n == 2:
            c_embedding = np.array([[-0.5, 0.0], [0.5, 0.0]])
            print(f"  簇 {cluster}: 2 个样本，放在 x 轴")
        elif n <= 5:
            try:
                c_embedding = PCA(n_components=2).fit_transform(c_features)
                print(f"  簇 {cluster}: {n} 个样本，使用 PCA 降维")
            except Exception:
                c_embedding = np.random.randn(n, 2) * 0.5
        else:
            cluster_reducer = umap.UMAP(
                n_neighbors=min(15, n - 1),
                min_dist=0.1, n_components=2,
                metric='euclidean', spread=5.0, random_state=42
            )
            c_embedding = cluster_reducer.fit_transform(c_features)
            print(f"  簇 {cluster}: {n} 个样本，使用 UMAP 降维")

        set_plot_font()
        plt.figure(figsize=(18, 18))

        # 先画散点（半透明底色）
        plt.scatter(
            c_embedding[:, 0], c_embedding[:, 1],
            c=[color_map[sl] for sl in c_source_labels],
            alpha=0.3, s=50
        )

        # 计算基础缩略图大小
        if len(c_embedding) > 1:
            avg_dist  = np.mean(pdist(c_embedding))
            base_size = max(20, min(thumbnail_size, avg_dist * 10))
        else:
            base_size = thumbnail_size

        # 在每个点上叠加图像缩略图
        for x, y, sl, img_path, sf in zip(
            c_embedding[:, 0], c_embedding[:, 1],
            c_source_labels, c_image_paths, c_size_factors
        ):
            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                ar   = w / h
                img_size = int(base_size * sf)
                if ar >= 1:
                    img = img.resize((img_size, max(1, int(img_size / ar))), Image.LANCZOS)
                else:
                    img = img.resize((max(1, int(img_size * ar)), img_size), Image.LANCZOS)

                arr = np.array(img)
                bc  = category_colors.get(sl, (0, 0, 0))
                bw  = 2
                arr[:bw,  :,  :] = bc
                arr[-bw:, :,  :] = bc
                arr[:,  :bw, :]  = bc
                arr[:, -bw:, :]  = bc

                imagebox = OffsetImage(Image.fromarray(arr), zoom=1.0)
                imagebox.image.axes = plt.gca()
                plt.gca().add_artist(
                    AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
                )
            except Exception as e:
                print(f"无法显示图像 {img_path}: {e}")

        # 图例
        existing_cats   = set(c_source_labels)
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=color_map[lbl],
                       markersize=18, label=category_names[lbl])
            for lbl in sorted(category_colors.keys())
            if lbl in existing_cats
        ]
        plt.legend(handles=legend_elements, loc='upper right',
                   prop={'weight': 'bold', 'size': 18},
                   labelspacing=0.5, borderpad=1.0, handletextpad=0.5, framealpha=0.9)

        plt.title(f"{method_name} - Cluster {cluster} ({len(cluster_indices)} samples)",
                  fontsize=32, fontweight='bold', pad=20)
        plt.xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
        plt.ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)
        plt.axis('equal')
        plt.margins(0.2)
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=18, pad=10)
        for tick in plt.gca().get_xticklabels():
            tick.set_fontweight('bold')
        for tick in plt.gca().get_yticklabels():
            tick.set_fontweight('bold')

        tif_path = os.path.join(out_dir, f"{method_name}_cluster{cluster}_images_improved.tif")
        plt.savefig(tif_path, format='tiff', dpi=150, pil_kwargs={'compression': 'none'})
        plt.close()

    print(f"  完成 {method_name} 的所有簇缩略图可视化")


# ==============================================================================
# 2-4. 参考域与候选域匹配可视化
# ==============================================================================

def visualize_matching_clusters(reference_results, candidate_results, matching_pairs,
                                save_dir,
                                category_names, category_colors, category_markers,
                                trained_umap=None, rule_matched_clusters=None):
    """
    可视化参考域与候选域的匹配簇（左右对比图）。

    参数:
        reference_results, candidate_results : 域结果字典
        matching_pairs                       : 匹配对列表 [(ref_idx, cand_idx, similarity), ...]
        save_dir                             : 保存目录
        category_names, category_colors, category_markers : 由 build_category_config 生成
        trained_umap                         : 可选，预训练 UMAP 模型
        rule_matched_clusters                : 可选，规则匹配字典

    返回:
        reducer: 训练好的 UMAP 模型
    """
    ref_features      = reference_results['features']
    ref_labels        = _to_int_list(reference_results['labels'])
    ref_source_labels = _to_int_list(reference_results['source_labels'])
    cand_features      = candidate_results['features']
    cand_labels        = _to_int_list(candidate_results['labels'])
    cand_source_labels = _to_int_list(candidate_results['source_labels'])

    n_ref  = reference_results['n_clusters']
    n_cand = candidate_results['n_clusters']

    ref_colors  = plt.cm.tab10(np.linspace(0, 1, n_ref))
    cand_colors = plt.cm.tab20(np.linspace(0, 1, n_cand))

    if trained_umap is None:
        reducer   = umap.UMAP(random_state=42)
        ref_umap  = reducer.fit_transform(ref_features)
        cand_umap = reducer.transform(cand_features)
    else:
        reducer   = trained_umap
        ref_umap  = trained_umap.transform(ref_features)
        cand_umap = trained_umap.transform(cand_features)

    plt.figure(figsize=(20, 10))

    # 参考域
    plt.subplot(1, 2, 1)
    for cluster in range(n_ref):
        cidx = [i for i, l in enumerate(ref_labels) if l == cluster]
        cemb = ref_umap[cidx]
        csl  = [ref_source_labels[i] for i in cidx]
        for cat in set(ref_source_labels):
            ii = [i for i, sl in enumerate(csl) if sl == cat]
            if ii:
                plt.scatter(cemb[ii, 0], cemb[ii, 1],
                            c=[ref_colors[cluster]],
                            marker=category_markers.get(cat, 'o'),
                            alpha=0.7, s=50,
                            edgecolors='black', linewidths=0.5)
    plt.title('Reference Domain Clusters', fontsize=32, fontweight='bold', pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=24, fontweight='bold', labelpad=20)
    plt.ylabel('UMAP Dimension 2', fontsize=24, fontweight='bold', labelpad=20)

    # 候选域
    plt.subplot(1, 2, 2)
    for cluster in range(n_cand):
        cidx = [i for i, l in enumerate(cand_labels) if l == cluster]
        cemb = cand_umap[cidx]
        csl  = [cand_source_labels[i] for i in cidx]
        for cat in set(cand_source_labels):
            ii = [i for i, sl in enumerate(csl) if sl == cat]
            if ii:
                plt.scatter(cemb[ii, 0], cemb[ii, 1],
                            c=cand_colors[cluster],
                            marker=category_markers.get(cat, 'o'),
                            alpha=0.7, s=50,
                            edgecolors='black', linewidths=0.5)
    plt.title('Candidate Domain Clusters', fontsize=32, fontweight='bold', pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=24, fontweight='bold', labelpad=20)
    plt.ylabel('UMAP Dimension 2', fontsize=24, fontweight='bold', labelpad=20)

    if rule_matched_clusters:
        rule_text = ", ".join(f"{k}: {len(v)}" for k, v in rule_matched_clusters.items())
        plt.figtext(0.5, 0.01, f'Matched Pairs: {len(matching_pairs)} ({rule_text})',
                    ha='center', fontsize=14)
    else:
        plt.figtext(0.5, 0.01, f'Matched Pairs: {len(matching_pairs)}',
                    ha='center', fontsize=14)

    plt.tight_layout()
    tif_path = os.path.join(save_dir, 'matching_visualization.tif')
    plt.savefig(tif_path, dpi=300, bbox_inches='tight',
                format='tiff', pil_kwargs={'compression': 'none'})
    plt.close()
    print(f"匹配可视化已保存到: {tif_path}")
    return reducer


# ==============================================================================
# 2-5. 相似度热图
# ==============================================================================

def plot_similarity_heatmap(similarity_matrix, x_labels, y_labels, title, save_path):
    """绘制相似度矩阵热图"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=x_labels, yticklabels=y_labels)
    plt.title(title, fontsize=28, fontweight='bold', pad=20)
    plt.xlabel("Candidate", fontsize=24, fontweight='bold', labelpad=20)
    plt.ylabel("Reference", fontsize=24, fontweight='bold', labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=18, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace('.png', '.tif'),
                dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
    plt.close()
    print(f"  热图已保存至: {save_path}")


# ==============================================================================
# 2-6. 单细胞 filter 之后的簇 UMAP 图
# ==============================================================================

def visualize_filtered_clusters(
    reference_results,
    candidate_results,
    matching_pairs,
    save_dir,
    category_names,
    category_colors,
    reference_umap=None,
    rule_matched_clusters=None,
    target_ref_indices=None,
    thumbnail_size=30,
    max_images=1000,
    passed_indices_dict=None,
):
    """
    可视化匹配的簇中保留的单细胞，保持与原始 UMAP 位置一致。

    参数:
        reference_results, candidate_results : 域结果字典
        matching_pairs                       : 匹配对列表
        save_dir                             : 保存目录
        category_names, category_colors      : 由 build_category_config 生成
        reference_umap                       : 参考 UMAP 嵌入（可选）
        rule_matched_clusters                : 规则匹配字典（可选）
        target_ref_indices                   : 目标参考簇索引（默认 [0, 5]）
        thumbnail_size                       : 基础缩略图大小
        max_images                           : 每个簇最大显示图像数
        passed_indices_dict                  : 外部传入的通过索引字典（可选）
                                               格式: {'rule1': [idx1, idx2, ...], ...}
    """
    print("为匹配的簇创建可视化（保持原始 UMAP 位置）...")

    color_map = _color_map_hex(category_colors)

    features      = candidate_results['features']
    labels        = _to_int_list(candidate_results['labels'])
    source_labels = _to_int_list(candidate_results['source_labels'])
    image_paths   = candidate_results['image_paths']

    cell_sims = cosine_similarity(features, reference_results['cluster_centers'])

    if target_ref_indices is None:
        target_ref_indices = [0, 5]

    # 确定需要处理的簇
    matched_clusters = []
    if rule_matched_clusters:
        for rule_name, clusters in rule_matched_clusters.items():
            for cluster in clusters:
                matched_clusters.append((cluster, rule_name))
    else:
        for ref_idx, cand_idx, _ in matching_pairs:
            if ref_idx == 5:
                matched_clusters.append((cand_idx, 'rule1'))
            elif ref_idx == 0:
                matched_clusters.append((cand_idx, 'rule2'))

    # 计算细胞大小
    if 'log_sizes' in candidate_results:
        log_sizes = np.asarray(candidate_results['log_sizes'], dtype=float)
    else:
        print("计算细胞大小...")
        cell_sizes = [count_non_background_pixels(p) for p in tqdm(image_paths, desc="计算细胞大小")]
        log_sizes  = np.log1p(np.array(cell_sizes))

    if np.max(log_sizes) > np.min(log_sizes):
        normalized         = (log_sizes - np.min(log_sizes)) / (np.max(log_sizes) - np.min(log_sizes))
        size_scale_factors = 0.5 + 1.5 * normalized
    else:
        size_scale_factors = np.ones_like(log_sizes)

    matched_vis_dir = os.path.join(save_dir, 'matched_clusters_visualization')
    os.makedirs(matched_vis_dir, exist_ok=True)

    for cluster_idx, rule_type in matched_clusters:
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_idx]

        # 筛选逻辑
        if passed_indices_dict is not None:
            rule_passed     = set(passed_indices_dict.get(rule_type, []))
            filtered_indices = [idx for idx in cluster_indices if idx in rule_passed]
        else:
            filtered_indices = [
                idx for idx in cluster_indices
                if np.argmax(cell_sims[idx]) in target_ref_indices
            ]

        if not filtered_indices:
            print(f"  簇 {cluster_idx} ({rule_type}) 没有匹配的细胞，跳过")
            continue

        if len(filtered_indices) > max_images:
            np.random.seed(42)
            selected_indices = list(np.random.choice(filtered_indices, max_images, replace=False))
        else:
            selected_indices = filtered_indices

        cluster_features       = features[cluster_indices]
        cluster_source_labels  = [source_labels[i] for i in cluster_indices]
        cluster_image_paths    = [image_paths[i]   for i in cluster_indices]
        cluster_size_factors   = [size_scale_factors[i] for i in cluster_indices]
        filtered_source_labels = [source_labels[i] for i in filtered_indices]

        # 降维
        n = len(cluster_features)
        if n == 1:
            cluster_embedding = np.array([[0.0, 0.0]])
        elif n == 2:
            cluster_embedding = np.array([[-0.5, 0.0], [0.5, 0.0]])
        elif n <= 5:
            try:
                cluster_embedding = PCA(n_components=2).fit_transform(cluster_features)
            except Exception:
                cluster_embedding = np.random.randn(n, 2) * 0.5
        else:
            cluster_reducer = umap.UMAP(
                n_neighbors=min(15, n - 1),
                min_dist=0.1, n_components=2,
                metric='euclidean', spread=5.0, random_state=42
            )
            cluster_embedding = cluster_reducer.fit_transform(cluster_features)

        # 构建 filtered_mask
        cluster_indices_arr = np.array(cluster_indices)
        filtered_mask       = np.zeros(len(cluster_indices), dtype=bool)
        for idx in filtered_indices:
            pos = np.where(cluster_indices_arr == idx)[0]
            if len(pos) > 0:
                filtered_mask[pos[0]] = True

        set_plot_font()
        plt.figure(figsize=(16, 16))

        # 所有细胞（灰色底）
        plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1],
                    c='lightgray', alpha=0.2, s=30)
        # 筛选后细胞（彩色）
        plt.scatter(
            cluster_embedding[filtered_mask, 0],
            cluster_embedding[filtered_mask, 1],
            c=[color_map[sl] for sl in filtered_source_labels],
            alpha=0.8, s=80
        )

        if len(cluster_embedding) > 1:
            avg_dist  = np.mean(pdist(cluster_embedding))
            base_size = max(20, min(thumbnail_size, avg_dist * 10))
        else:
            base_size = thumbnail_size

        # 只为筛选后的细胞显示图像
        for i, idx in enumerate(np.where(filtered_mask)[0]):
            x, y     = cluster_embedding[idx, 0], cluster_embedding[idx, 1]
            sl       = cluster_source_labels[idx]
            img_path = cluster_image_paths[idx]
            sf       = cluster_size_factors[idx]
            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                ar   = w / h
                img_size = int(base_size * sf)
                if ar >= 1:
                    img = img.resize((img_size, max(1, int(img_size / ar))), Image.LANCZOS)
                else:
                    img = img.resize((max(1, int(img_size * ar)), img_size), Image.LANCZOS)
                arr = np.array(img)
                bc  = category_colors.get(sl, (0, 0, 0))
                bw  = 2
                arr[:bw,  :,  :] = bc
                arr[-bw:, :,  :] = bc
                arr[:,  :bw, :]  = bc
                arr[:, -bw:, :]  = bc
                imagebox = OffsetImage(Image.fromarray(arr), zoom=1.0)
                imagebox.image.axes = plt.gca()
                plt.gca().add_artist(
                    AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
                )
            except Exception as e:
                print(f"无法显示图像 {img_path}: {e}")

        existing_cats   = set(filtered_source_labels)
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=color_map[lbl],
                       markersize=16, label=category_names[lbl])
            for lbl in sorted(category_colors.keys())
            if lbl in existing_cats
        ]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='lightgray',
                       markersize=16, label='filtered cells')
        )
        plt.legend(handles=legend_elements, loc='upper right',
                   prop={'weight': 'bold', 'size': 16},
                   labelspacing=0.5, borderpad=1.0, handletextpad=0.5, framealpha=0.9)

        plt.title(
            f"{rule_type} - Cluster {cluster_idx} Matched Cells "
            f"({len(filtered_indices)}/{len(cluster_indices)} cells)",
            fontsize=32, fontweight='bold', pad=20
        )
        plt.xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
        plt.ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)
        plt.axis('equal')
        plt.margins(0.2)
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=18, pad=10)
        for tick in plt.gca().get_xticklabels():
            tick.set_fontweight('bold')
        for tick in plt.gca().get_yticklabels():
            tick.set_fontweight('bold')

        tif_path = os.path.join(
            matched_vis_dir,
            f"{rule_type}_cluster{cluster_idx}_matched_cells.tif"
        )
        plt.savefig(tif_path, dpi=150, bbox_inches='tight',
                    format='tiff', pil_kwargs={'compression': 'none'})
        plt.close()

    print("  匹配簇可视化完成")
    return matched_vis_dir
