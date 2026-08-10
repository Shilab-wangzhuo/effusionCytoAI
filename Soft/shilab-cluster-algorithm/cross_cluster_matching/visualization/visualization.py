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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


# 1. 定义可视化功能函数中使用的类别名称颜色和marker形状
category_names = {
    0: 'Candidate',
    1: 'LUAD',
    2: 'LUSC',
    3: 'SCLC',
    4: 'Benign',
    5: 'Malignant'
}

category_colors = {
    0: (40, 120, 181), # Candidate: 深蓝色
    1: (255, 136, 132), # LUAD: 浅粉色
    2: (200, 36, 35), # LUSC: 深红色
    3: (255, 165, 0), # SCLC: 橙色
    4: (154, 201, 219), # Benign: 浅蓝色
    5: (185, 89, 184)  # Malignant:紫色表示恶性
}

category_markers = {
    0: 's',  # Candidate: 正方形 (Square) - 中立、稳定，与圆和三角完全区分
    1: 'v',  # LUAD: 倒三角形 (Triangle Down) - 恶性家族成员
    2: '<',  # LUSC: 左三角形 (Triangle Left) - 恶性家族成员
    3: '>',  # SCLC: 右三角形 (Triangle Right) - 恶性家族成员
    4: 'o',  # Benign: 圆形 (Circle) - 平滑、安全，与恶性的尖锐感形成最大反差
    5: '^'   # Malignant: 正三角形 (Triangle Up) - 恶性家族的基准符号
    }

def set_plot_font():
    """设置绘图字体参数"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight'] = 'bold'


# 2.定义可视化功能函数

# 计算细胞大小（图像中非(225,225,225)像素点的数量）
def count_non_background_pixels(image_path):
    """
    计算图像中非(225,225,225)背景像素的数量，作为细胞大小的估计
    
    参数:
    image_path: 图像路径
    
    返回:
    非背景像素的数量
    """
    try:
        # 读取图像
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # 计算非(225,225,225)像素的数量
        # 创建背景掩码 (225,225,225)
        background_mask = np.all(img_array == [225, 225, 225], axis=2)
        # 计算非背景像素数量
        non_background_count = np.sum(~background_mask)
        
        return non_background_count
    except Exception as e:
        print(f"计算图像 {image_path} 的非背景像素数量时出错: {e}")
        return 0

# 2-1. 统计每个簇中各类别的数量（用来画reference域的柱状图）
def analyze_cluster_composition(labels, source_labels, image_paths, method_name, save_dir):
    """
    统计每个簇中各类别的数量
    
    参数:
    labels: 聚类标签
    source_labels: 源标签 (0: candidate, 1: LUAD, 2: LUSC, 3: SCLC, 4: Benign, 5: Malignant)
    image_paths: 图像路径
    method_name: 聚类方法名称
    save_dir: 保存目录
    
    返回:
    每个簇的类别分布字典
    """

    # 将RGB颜色转换为matplotlib可用的格式
    color_map = {
        i: f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for i, c in category_colors.items()
    }
    
    # 获取所有簇标签
    unique_clusters = sorted(set(labels))
    
    # 创建结果字典
    cluster_composition = {}
    
    # 创建DataFrame来存储每个图像的聚类结果
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'source_label': source_labels,
        'cluster': labels
    })
    
    # 添加类别名称列 - 使用字典形式的category_names
    results_df['category'] = results_df['source_label'].map(category_names)
    
    # 获取数据中实际存在的类别
    existing_categories = set(results_df['source_label'].unique())
    
    # 对于每个簇，统计各类别的数量
    for cluster in unique_clusters:
        # 获取该簇的所有样本
        cluster_samples = results_df[results_df['cluster'] == cluster]
        
        # 统计各类别的数量 - 只包含存在的类别
        category_counts = cluster_samples['source_label'].value_counts().to_dict()
        
        # 获取所有可能的类别（从category_names字典中获取）
        all_possible_categories = set(category_names.keys())
        
        # 存储结果 - 包含所有可能的类别，但只显示存在的类别
        cluster_composition[cluster] = {
            'total': len(cluster_samples),
            'categories': {category_names[i]: category_counts.get(i, 0) for i in all_possible_categories if i in existing_categories}
        }
    
    # 创建一个总结表格
    summary_data = []
    
    for cluster, data in cluster_composition.items():
        row = {'Cluster': cluster, 'Total': data['total']}
        row.update(data['categories'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    # 保存summary_df为CSV文件
    csv_path = os.path.join(save_dir, f"{method_name}_cluster_composition.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"聚类组成摘要已保存到: {csv_path}")

    set_plot_font()
    
    # 创建带有外部图例空间的图形
    fig = plt.figure(figsize=(18, 12))
    
    # 创建主绘图区域，留出右侧空间给图例
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # [left, bottom, width, height]
    
    # 准备数据
    clusters = summary_df['Cluster'].tolist()
    
    # 使用指定的颜色创建堆叠条形图
    bottom = np.zeros(len(clusters))
    
    # 创建图例元素列表
    legend_elements = []
    
    # 只为存在的类别创建条形图和图例，使用字典形式的category_names
    for i in sorted(category_names.keys()):
        category = category_names[i]
        if i in existing_categories and category in summary_df.columns:
            values = summary_df[category].values
            ax.bar(clusters, values, bottom=bottom, color=color_map[i])
            bottom += values
            
            # 为图例添加元素
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, color=color_map[i], label=category)
            )
    
    # 设置主图的标签和标题
    ax.set_xlabel('Cluster', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('Number of Samples', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_title(f'{method_name} - Cluster Composition', fontsize=32, fontweight='bold', pad=20)
    
    # 设置x轴刻度
    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters, rotation=90 if len(clusters) > 10 else 0)
    ax.set_xlim(-0.6, len(clusters) - 0.4)
    
    # 设置刻度样式
    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 创建图例区域在图表右侧
    legend_ax = fig.add_axes([0.76, 0.5, 0.1, 0.4])  # [left, bottom, width, height]
    legend_ax.axis('off')  # 隐藏坐标轴
    
    # 添加图例
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
    
    # 保存图像
    # plt.savefig(os.path.join(save_dir, f"{method_name}_cluster_composition.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{method_name}_cluster_composition.tif"), dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
    plt.close()

    return cluster_composition

# 2-2. 画reference domain和candidate domain的全局UMAP可视化--颜色区分簇，点形状区分类别
# def visualize_all_clusters_without_images(features, labels, source_labels, method_name, save_dir, trained_umap=None):
#     """
#     创建一个全局UMAP可视化，显示所有簇的分布，但不包含缩略图
#     - 不同簇使用不同颜色
#     - 不同类别使用不同形状的标记
    
#     参数:
#     features: 特征矩阵
#     labels: 聚类标签
#     source_labels: 源标签（细胞类型）
#     method_name: 方法名称，用于标题和文件名
#     save_dir: 保存目录
#     trained_umap: 可选，预先训练好的UMAP模型，如果提供则使用该模型进行转换而不是重新训练
#     """
#     print(f"为{method_name}创建不带缩略图的全局UMAP可视化...")
    
#     # 使用UMAP进行可视化降维
#     if trained_umap is None:
#         # 如果没有提供训练好的UMAP模型，则创建并训练一个新模型
#         reducer = umap.UMAP(random_state=42)
#         embedding = reducer.fit_transform(features)
#     else:
#         # 如果提供了训练好的UMAP模型，则直接使用该模型进行转换
#         embedding = trained_umap.transform(features)
    
#     # 获取所有簇标签
#     unique_clusters = sorted(set(labels))
    
#     # 簇颜色映射 - 使用不同的颜色代表不同的簇
#     cluster_colors = cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    
#     set_plot_font()
    
#     plt.figure(figsize=(14, 12))
    
#     # 为每个簇和类别组合创建散点图
#     for cluster_idx, cluster in enumerate(unique_clusters):
#         # 获取该簇的样本索引
#         cluster_indices = np.where(labels == cluster)[0]
        
#         # 获取该簇的嵌入和源标签
#         cluster_embedding = embedding[cluster_indices]
#         cluster_source_labels = [source_labels[i] for i in cluster_indices]
        
#         # 为每个类别绘制点
#         for category in set(source_labels):
#             # 获取该簇中该类别的样本索引
#             category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            
#             if category_indices:
#                 # 获取该簇中该类别的样本嵌入
#                 category_embedding = cluster_embedding[category_indices]
                
#                 # 绘制散点图
#                 plt.scatter(
#                     category_embedding[:, 0],
#                     category_embedding[:, 1],
#                     c=[cluster_colors[cluster_idx]],
#                     marker=category_markers.get(category, 'o'),
#                     label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
#                     alpha=0.7,
#                     s=50,
#                     edgecolors='black',
#                     linewidths=0.5
#                 )
    
#     # 创建自定义图例
#     # 1. 簇的颜色图例
#     cluster_legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', 
#                  markerfacecolor=cluster_colors[i],
#                  markersize=16, label=f'Cluster {cluster}')
#         for i, cluster in enumerate(unique_clusters)
#     ]
    
#     # 2. 类别的标记图例
#     category_legend_elements = [
#         plt.Line2D([0], [0], marker=marker, color='black', 
#                  markersize=16, label=category_names[cat])
#         for cat, marker in category_markers.items()
#         if cat in source_labels
#     ]
    
#     # 添加两个图例
#     legend1 = plt.legend(handles=cluster_legend_elements, loc='upper right', 
#                     title="Clusters",title_fontsize=14, bbox_to_anchor=(1.0, 1.0), prop={'weight': 'bold', 'size': 14},
#                     labelspacing=0.5,  # <--- 修改点: 增加图例条目之间的垂直间距 (默认约0.5)
#                     borderpad=1.0,     # <--- 修改点: 增加图例边框内部的留白
#                     handletextpad=0.5, # <--- 修改点: 增加图标和文字之间的距离
#                     framealpha=0.9     # <--- (可选) 增加背景不透明度，防止遮挡看不清)
#     )
#     plt.gca().add_artist(legend1)  # 确保第一个图例不会被清除

#     legend2 = plt.legend(handles=category_legend_elements, loc='lower right', 
#                     title="Categories", title_fontsize=14, bbox_to_anchor=(1.0, 0.0), prop={'weight': 'bold', 'size': 14},
#                     labelspacing=0.5,  # <--- 修改点: 增加图例条目之间的垂直间距 (默认约0.5)
#                     borderpad=1.0,     # <--- 修改点: 增加图例边框内部的留白
#                     handletextpad=0.5, # <--- 修改点: 增加图标和文字之间的距离
#                     framealpha=0.9     # <--- (可选) 增加背景不透明度，防止遮挡看不清)
#                     )
#     plt.gca().add_artist(legend2)
    
#     # 设置标题和轴标签
#     plt.title(f"{method_name} - All Clusters Distribution", fontsize=32, fontweight='bold', pad=20)
#     plt.xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold',labelpad=20)
#     plt.ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold',labelpad=20)
    
#     # 调整坐标轴范围
#     plt.tight_layout()
#     plt.tick_params(axis='both', which='major', labelsize=18,pad=10)
#     for tick in plt.gca().get_xticklabels():
#         tick.set_fontweight('bold')
#     for tick in plt.gca().get_yticklabels():
#         tick.set_fontweight('bold')
    
#     # 保存图像
#     os.makedirs(os.path.join(save_dir, 'cluster_images'), exist_ok=True)
#     plt.savefig(os.path.join(save_dir, 'cluster_images', f"{method_name}_all_clusters_distribution.png"), 
#                dpi=300, bbox_inches='tight')
#     plt.savefig(os.path.join(save_dir, 'cluster_images', f"{method_name}_all_clusters_distribution.tif"), 
#                dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
#     plt.close()
    
#     print(f"  完成{method_name}的不带缩略图的全局可视化")
    
#     # 返回训练好的UMAP模型，以便其他函数使用
#     if trained_umap is None:
#         return reducer
#     else:
#         return trained_umap
# 2-2. 画reference domain和candidate domain的全局UMAP可视化--颜色区分簇，点形状区分类别
def visualize_all_clusters_without_images(features, labels, source_labels, method_name, save_dir,
                                          trained_umap=None, output_format="tif"):
    """
    创建一个全局UMAP可视化，显示所有簇的分布，但不包含缩略图
    - 不同簇使用不同颜色
    - 不同类别使用不同形状的标记
    
    参数:
    features: 特征矩阵
    labels: 聚类标签
    source_labels: 源标签（细胞类型）
    method_name: 方法名称，用于标题和文件名
    save_dir: 保存目录
    trained_umap: 可选，预先训练好的UMAP模型，如果提供则使用该模型进行转换而不是重新训练
    """
    print(f"为{method_name}创建不带缩略图的全局UMAP可视化...")
    
    # 使用UMAP进行可视化降维
    if trained_umap is None:
        # 如果没有提供训练好的UMAP模型，则创建并训练一个新模型
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        # 如果提供了训练好的UMAP模型，则直接使用该模型进行转换
        embedding = trained_umap.transform(features)
    
    # 获取所有簇标签
    unique_clusters = sorted(set(labels))
    
    # 簇颜色映射 - 使用不同的颜色代表不同的簇
    cluster_colors = cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    
    set_plot_font()
    
    # 创建一个带有外部图例空间的图形
    fig = plt.figure(figsize=(18, 12))  # 增加图形宽度，为图例留出空间
    
    # 创建主绘图区域，留出右侧空间给图例
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # [left, bottom, width, height]
    
    # 为每个簇和类别组合创建散点图
    for cluster_idx, cluster in enumerate(unique_clusters):
        # 获取该簇的样本索引
        cluster_indices = np.where(labels == cluster)[0]
        
        # 获取该簇的嵌入和源标签
        cluster_embedding = embedding[cluster_indices]
        cluster_source_labels = [source_labels[i] for i in cluster_indices]
        
        # 为每个类别绘制点
        for category in set(source_labels):
            # 获取该簇中该类别的样本索引
            category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            
            if category_indices:
                # 获取该簇中该类别的样本嵌入
                category_embedding = cluster_embedding[category_indices]
                
                # 绘制散点图
                ax.scatter(
                    category_embedding[:, 0],
                    category_embedding[:, 1],
                    c=[cluster_colors[cluster_idx]],
                    marker=category_markers.get(category, 'o'),
                    label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5
                )
    
    # 创建自定义图例
    # 1. 簇的颜色图例
    cluster_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                 markerfacecolor=cluster_colors[i],
                 markersize=16, label=f'Cluster {cluster}')
        for i, cluster in enumerate(unique_clusters)
    ]
    
    # 2. 类别的标记图例
    category_legend_elements = [
        plt.Line2D([0], [0], marker=marker, color='black', 
                 markersize=16, label=category_names[cat])
        for cat, marker in category_markers.items()
        if cat in source_labels
    ]
    
    # 创建图例区域在图表右侧
    legend_ax = fig.add_axes([0.76, 0.5, 0.1, 0.4])  # [left, bottom, width, height]
    legend_ax.axis('off')  # 隐藏坐标轴
    
    # 添加簇图例到图例区域上半部分
    legend1 = legend_ax.legend(
        handles=cluster_legend_elements,  # 图例元素列表，每个元素代表一个要在图例中显示的项目
        title="Clusters",                 # 图例的标题文本
        title_fontsize=18,                # 图例标题的字体大小
        prop={'weight': 'bold', 'size': 18},  # 图例文本的属性：粗体，字号14
        labelspacing=0.8,                 # 图例项目之间的垂直间距
        borderpad=1.0,                    # 图例边框与内容之间的内边距
        handletextpad=0.5,                # 图例标记(符号)与对应文本之间的水平间距
        frameon=True,                     # 是否显示图例的边框
        loc='center left'                 # 图例在其容器(legend_ax)中的位置
    )
    legend_ax.add_artist(legend1)
    
    # 创建第二个图例区域
    legend_ax2 = fig.add_axes([0.76, 0.1, 0.1, 0.4])  # [left, bottom, width, height]
    legend_ax2.axis('off')  # 隐藏坐标轴
    
    # 添加类别图例到图例区域下半部分
    legend2 = legend_ax2.legend(handles=category_legend_elements, 
                    title="Categories", title_fontsize=18, 
                    prop={'weight': 'bold', 'size': 18},
                    labelspacing=0.8,  
                    borderpad=1.0,     
                    handletextpad=0.5,
                    frameon=True,
                    loc='center left')
    legend_ax2.add_artist(legend2)
    
    # 设置主图标题和轴标签
    ax.set_title(f"{method_name} - All Clusters Distribution", fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)
    ax.margins(x=0.08, y=0.08)
    ax.set_aspect('equal', adjustable='datalim')
    
    # 调整坐标轴刻度
    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 保存图像
    os.makedirs(os.path.join(save_dir, 'cluster_images'), exist_ok=True)
    # plt.savefig(os.path.join(save_dir, 'cluster_images', f"{method_name}_all_clusters_distribution.png"), 
    #            dpi=300, bbox_inches='tight')
    output_path = os.path.join(save_dir, 'cluster_images', f"{method_name}_all_clusters_distribution.{output_format}")
    save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.20}
    if output_format == 'tif':
        save_kwargs.update({'format': 'tiff', 'pil_kwargs': {'compression': 'none'}})
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    
    print(f"  完成{method_name}的不带缩略图的全局可视化")
    
    # 返回训练好的UMAP模型，以便其他函数使用
    if trained_umap is None:
        return reducer
    else:
        return trained_umap


# 2-2-1. 画reference domain和candidate domain的全局UMAP可视化--颜色区分类别，点全是圆点
def visualize_all_clusters_without_images_1(features, labels, source_labels, method_name, save_dir,
                                            trained_umap=None, output_format="tif"):
    """
    创建一个全局UMAP可视化，显示所有簇的分布，但不包含缩略图
    - 不同类别使用不同颜色（根据category_colors映射）
    - 所有点使用相同形状（圆形）
    
    参数:
    features: 特征矩阵
    labels: 聚类标签
    source_labels: 源标签（细胞类型）
    method_name: 方法名称，用于标题和文件名
    save_dir: 保存目录
    trained_umap: 可选，预先训练好的UMAP模型，如果提供则使用该模型进行转换而不是重新训练
    """
    print(f"为{method_name}创建不带缩略图的全局UMAP可视化...")
    
    # 使用UMAP进行可视化降维
    if trained_umap is None:
        # 如果没有提供训练好的UMAP模型，则创建并训练一个新模型
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        # 如果提供了训练好的UMAP模型，则直接使用该模型进行转换
        embedding = trained_umap.transform(features)
    
    # 获取所有簇标签
    unique_clusters = sorted(set(labels))
    
    # 将RGB颜色转换为matplotlib可用的格式
    color_map = {
        i: f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for i, c in category_colors.items()
    }
    
    set_plot_font()
    
    fig = plt.figure(figsize=(18, 12))
    
    # 创建主绘图区域，留出右侧空间给图例
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])  # [left, bottom, width, height]
    
    # 为每个簇创建散点图
    for cluster in unique_clusters:
        # 获取该簇的样本索引
        cluster_indices = np.where(labels == cluster)[0]
        
        # 获取该簇的嵌入和源标签
        cluster_embedding = embedding[cluster_indices]
        cluster_source_labels = [source_labels[i] for i in cluster_indices]
        
        # 为每个类别绘制点
        for category in set(source_labels):
            # 获取该簇中该类别的样本索引
            category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            
            if category_indices:
                # 获取该簇中该类别的样本嵌入
                category_embedding = cluster_embedding[category_indices]
                
                # 绘制散点图 - 使用统一的圆形标记，但颜色根据类别区分
                ax.scatter(
                    category_embedding[:, 0],
                    category_embedding[:, 1],
                    c=color_map[category],  # 根据类别设置颜色
                    marker='o',  # 统一使用圆形标记
                    label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5
                )
    
    # 创建自定义图例 - 只需要类别的颜色图例
    category_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                 markerfacecolor=color_map[cat],
                 markersize=16, label=category_names[cat])
        for cat in sorted(category_colors.keys())
        if cat in source_labels
    ]

    # 创建图例区域在图表右侧
    legend_ax = fig.add_axes([0.76, 0.5, 0.1, 0.4])  # [left, bottom, width, height]
    legend_ax.axis('off')  # 隐藏坐标轴

    # 添加类别图例
    legend = legend_ax.legend(
                handles=category_legend_elements, 
                loc='center left', 
                title="Categories", 
                title_fontsize=18, 
                prop={'weight': 'bold', 'size': 18},
                labelspacing=0.5,  # 增加图例条目之间的垂直间距
                borderpad=1.0,     # 增加图例边框内部的留白
                handletextpad=0.5, # 增加图标和文字之间的距离
                framealpha=0.9     # 增加背景不透明度，防止遮挡看不清
                )
    legend_ax.add_artist(legend)

    # 设置标题和轴标签
    ax.set_title(f"{method_name} - All Clusters Distribution", fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold', labelpad=20)
    
    # 调整坐标轴范围
    
    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # 保存图像
    os.makedirs(os.path.join(save_dir, 'cluster_images'), exist_ok=True)
    # plt.savefig(os.path.join(save_dir, 'cluster_images', f"{method_name}_all_clusters_distribution_1.png"), 
    #            dpi=300, bbox_inches='tight')
    ax.margins(x=0.08, y=0.08)
    ax.set_aspect('equal', adjustable='datalim')
    output_path = os.path.join(save_dir, 'cluster_images', f"{method_name}_all_clusters_distribution_1.{output_format}")
    save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.20}
    if output_format == 'tif':
        save_kwargs.update({'format': 'tiff', 'pil_kwargs': {'compression': 'none'}})
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    
    print(f"  完成{method_name}的不带缩略图的全局可视化")
    
    # 返回训练好的UMAP模型，以便其他函数使用
    if trained_umap is None:
        return reducer
    else:
        return trained_umap

# 2-3. 画每一个簇的umap图（缩略图大小反映细胞的实际大小）
def improved_visualize_clusters_with_images(features, labels, source_labels, image_paths, method_name, save_dir,
                                            log_sizes=None, thumbnail_size=30, max_images=1000,
                                            output_format="tif"):
    """
    为每个簇创建UMAP可视化，并在点上显示图像
    - 直接在图像上添加边框
    - 缩略图大小反映细胞的实际大小（对数变换后）
    - 保持原始图像的长宽比
    
    参数:
    features: 特征矩阵
    labels: 聚类标签
    source_labels: 源标签（类别）
    image_paths: 图像路径
    method_name: 聚类方法名称
    save_dir: 保存目录
    thumbnail_size: 基础缩略图大小
    max_images: 每个簇最大显示的图像数量
    """
    print(f"为{method_name}的每个簇创建可视化...")

    if log_sizes is None:
        # 计算每个图像的细胞大小（非白像素点数量）
        print("计算细胞大小...")
        cell_sizes = []
        for img_path in tqdm(image_paths, desc="计算细胞大小"):
            size = count_non_background_pixels(img_path)
            cell_sizes.append(size)
        
        # 对细胞大小进行对数变换
        log_sizes = np.log1p(np.array(cell_sizes))
    
    # 归一化对数变换后的细胞大小，用于缩放图像
    if np.max(log_sizes) > np.min(log_sizes):
        normalized_log_sizes = (log_sizes - np.min(log_sizes)) / (np.max(log_sizes) - np.min(log_sizes))
        # 将归一化值缩放到合理的图像大小范围（0.5到2倍基础大小）
        size_scale_factors = 0.5 + 1.5 * normalized_log_sizes
    else:
        size_scale_factors = np.ones_like(log_sizes)
    
    # 获取所有簇标签
    unique_clusters = sorted(set(labels))
    
    # 为每个簇创建可视化
    for cluster in unique_clusters:
        # 获取该簇的样本索引
        cluster_indices = np.where(labels == cluster)[0]
        
        # 如果样本太多，随机采样
        if len(cluster_indices) > max_images:
            np.random.seed(42)
            selected_indices = np.random.choice(cluster_indices, max_images, replace=False)
        else:
            selected_indices = cluster_indices
        
        # 获取该簇的特征、源标签和图像路径
        cluster_features = features[selected_indices]
        cluster_source_labels = [source_labels[i] for i in selected_indices]
        cluster_image_paths = [image_paths[i] for i in selected_indices]
        cluster_size_factors = [size_scale_factors[i] for i in selected_indices]
        
        # 修改这里：根据样本数量选择不同的降维方法
        if len(cluster_features) == 1:
            # 如果只有一个样本，直接将其放在原点
            cluster_embedding = np.array([[0.0, 0.0]])
            print(f"  聚类 {cluster} 只有1个样本，将其放在原点")
        elif len(cluster_features) == 2:
            # 如果只有两个样本，将它们放在x轴上相距为1的位置
            cluster_embedding = np.array([[-0.5, 0.0], [0.5, 0.0]])
            print(f"  聚类 {cluster} 只有2个样本，将它们放在x轴上")
        elif len(cluster_features) <= 5:
            # 如果样本数量小于等于5，使用PCA
            try:
                from sklearn.decomposition import PCA
                cluster_embedding = PCA(n_components=2).fit_transform(cluster_features)
                print(f"  聚类 {cluster} 有{len(cluster_features)}个样本，使用PCA降维")
            except Exception as e:
                print(f"  聚类 {cluster} PCA降维失败: {e}，使用随机投影")
                # 如果PCA失败，使用随机投影
                cluster_embedding = np.random.randn(len(cluster_features), 2) * 0.5
        else:
            # 使用UMAP进行可视化降维
            cluster_reducer = umap.UMAP(
                n_neighbors=min(15, len(cluster_features)-1),  # 避免n_neighbors大于样本数
                min_dist=0.1,
                n_components=2,
                metric='euclidean',
                spread=5.0,    # 大幅增加扩散参数，使点更分散
                random_state=42
            )
            cluster_embedding = cluster_reducer.fit_transform(cluster_features)
            print(f"  聚类 {cluster} 有{len(cluster_features)}个样本，使用UMAP降维")
        
        set_plot_font()

        # 创建图像 - 减小图像尺寸
        plt.figure(figsize=(16, 16))  # 从(24, 24)减小到(16, 16)
        
        # 首先绘制该簇的点
        plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], 
                   c=[f'#{category_colors[sl][0]:02x}{category_colors[sl][1]:02x}{category_colors[sl][2]:02x}' 
                     for sl in cluster_source_labels], 
                   alpha=0.3, s=50)
        
        # 计算点之间的平均距离，用于缩放图像大小
        if len(cluster_embedding) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(cluster_embedding)
            avg_distance = np.mean(distances)
            # 根据平均距离调整基础缩略图大小
            base_size = min(thumbnail_size, avg_distance * 10)
            base_size = max(base_size, 20)
        else:
            base_size = thumbnail_size
        
        # 加载并显示图像
        for i, (x, y, source_label, img_path, size_factor) in enumerate(zip(
            cluster_embedding[:, 0], cluster_embedding[:, 1], 
            cluster_source_labels, cluster_image_paths, cluster_size_factors)):
            
            try:
                # 加载图像
                img = Image.open(img_path).convert('RGB')
                
                # 获取原始图像的宽高比
                width, height = img.size
                aspect_ratio = width / height
                
                # 根据细胞大小（对数变换后）调整图像大小，同时保持长宽比
                img_size = int(base_size * size_factor)
                
                if aspect_ratio >= 1:  # 宽图
                    new_width = img_size
                    new_height = int(img_size / aspect_ratio)
                else:  # 高图
                    new_height = img_size
                    new_width = int(img_size * aspect_ratio)
                
                # 调整图像大小，保持长宽比
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # 将PIL图像转换为numpy数组
                img_array = np.array(img)
                
                # 获取边框颜色
                border_color = category_colors.get(source_label, (0, 0, 0))
                
                # 直接在图像上添加边框
                border_width = 2
                # 上边框
                img_array[:border_width, :, :] = border_color
                # 下边框
                img_array[-border_width:, :, :] = border_color
                # 左边框
                img_array[:, :border_width, :] = border_color
                # 右边框
                img_array[:, -border_width:, :] = border_color
                
                # 将numpy数组转换回PIL图像
                bordered_img = Image.fromarray(img_array)
                
                # 创建OffsetImage
                imagebox = OffsetImage(bordered_img, zoom=1.0)
                imagebox.image.axes = plt.gca()
                
                # 创建AnnotationBbox - 不带边框
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
                plt.gca().add_artist(ab)
                
            except Exception as e:
                print(f"无法显示图像 {img_path}: {e}")
                continue
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', 
                     markerfacecolor=f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}', 
                     markersize=18, label=category_names[label])
            for label, c in category_colors.items()
            if label in cluster_source_labels
        ]
        plt.legend(handles=legend_elements,loc='upper right', prop={'weight': 'bold', 'size': 18},
                    labelspacing=0.5,  # <--- 修改点: 增加图例条目之间的垂直间距 (默认约0.5)
                    borderpad=1.0,     # <--- 修改点: 增加图例边框内部的留白
                    handletextpad=0.5, # <--- 修改点: 增加图标和文字之间的距离
                    framealpha=0.9     # <--- (可选) 增加背景不透明度，防止遮挡看不清)
                    )
        
        # 设置标题和轴标签
        plt.title(f"{method_name} - Cluster {cluster} ({len(cluster_indices)} samples)", fontsize=32, fontweight='bold', pad=20)
        plt.xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold',labelpad=20)
        plt.ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold',labelpad=20)
        
        # 调整坐标轴范围以适应图像
        plt.axis('equal')
        
        # 自动调整坐标轴，确保所有图像都可见
        plt.margins(0.2)
        
        # 增加边距，确保图像不会被裁剪
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=18,pad=10)
        for tick in plt.gca().get_xticklabels():
            tick.set_fontweight('bold')
        for tick in plt.gca().get_yticklabels():
            tick.set_fontweight('bold')
        
        # 保存图像 - 减小DPI以减小文件大小，但保持足够的清晰度
        os.makedirs(os.path.join(save_dir, 'cluster_images'), exist_ok=True)
        # plt.savefig(os.path.join(save_dir, 'cluster_images', f"{method_name}_cluster{cluster}_images_improved.png"), 
        #            dpi=150, bbox_inches='tight')  # 从300减小到150 DPI
        output_path = os.path.join(save_dir, 'cluster_images', f"{method_name}_cluster{cluster}_images_improved.{output_format}")
        save_kwargs = {'dpi': 150, 'bbox_inches': 'tight'}
        if output_format == 'tif':
            save_kwargs.update({'format': 'tiff', 'pil_kwargs': {'compression': 'none'}})
        plt.savefig(output_path, **save_kwargs)
        plt.close()
    
    print(f"  完成{method_name}的可视化")

# 2-4. 可视化reference domain和candidate domain，并用颜色进行匹配（需要修改）
def visualize_matching_clusters(reference_results, candidate_results, matching_pairs, save_dir, 
                               trained_umap=None, rule_matched_clusters=None, output_format="tif"):
    """
    可视化匹配簇 - 修改版，适用于任意数量的匹配规则
    
    参数:
    reference_results: 参考域结果字典
    candidate_results: 候选域结果字典
    matching_pairs: 匹配对列表，每个元素为(ref_idx, cand_idx, similarity)
    save_dir: 保存结果的目录
    trained_umap: 预训练的UMAP模型
    rule_matched_clusters: 字典，键为规则名称（如'rule1', 'rule2', 'rule3'），值为该规则匹配的簇列表
    """
    # 提取参考域和候选域的特征和标签
    reference_embedding = reference_results['features']
    reference_labels = reference_results['labels']
    reference_source_labels = reference_results['source_labels']
    
    candidate_embedding = candidate_results['features']
    candidate_labels = candidate_results['labels']
    candidate_source_labels = candidate_results['source_labels']
    
    # 获取簇数量
    n_ref_clusters = reference_results['n_clusters']
    n_cand_clusters = candidate_results['n_clusters']
    
    # 创建颜色映射
    ref_colors = plt.cm.tab10(np.linspace(0, 1, n_ref_clusters))
    cand_colors = plt.cm.tab20(np.linspace(0, 1, n_cand_clusters))
    
    # 使用UMAP进行降维可视化
    if trained_umap is None:
        reducer = umap.UMAP(random_state=42)
        reference_umap = reducer.fit_transform(reference_embedding)
        candidate_umap = reducer.transform(candidate_embedding)
    else:
        reference_umap = trained_umap.transform(reference_embedding)
        candidate_umap = trained_umap.transform(candidate_embedding)
    
    # 创建图形
    plt.figure(figsize=(20, 10))
    
    # 可视化参考域
    plt.subplot(1, 2, 1)
    
    # 为每个簇和类别组合创建散点图
    for cluster in range(n_ref_clusters):
        # 获取该簇的样本索引
        cluster_indices = np.where(reference_labels == cluster)[0]
        
        # 获取该簇的嵌入和源标签
        cluster_embedding = reference_umap[cluster_indices]
        cluster_source_labels = [reference_source_labels[i] for i in cluster_indices]
        
        # 为每个类别绘制点
        for category in set(reference_source_labels):
            # 获取该簇中该类别的样本索引
            category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            
            if category_indices:
                # 获取该簇中该类别的样本嵌入
                category_embedding = cluster_embedding[category_indices]
                
                # 绘制散点图
                plt.scatter(
                    category_embedding[:, 0],
                    category_embedding[:, 1],
                    c=[ref_colors[cluster]],
                    marker=category_markers.get(category, 'o'),
                    label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5
                )
    
    plt.title('Reference Domain Clusters', fontsize=32, fontweight='bold', pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=24, fontweight='bold',labelpad=20)
    plt.ylabel('UMAP Dimension 2', fontsize=24, fontweight='bold',labelpad=20)
    
    # 可视化candidate域
    plt.subplot(1, 2, 2)
    
    # 为每个簇和类别组合创建散点图
    for cluster in range(n_cand_clusters):
        # 获取该簇的样本索引
        cluster_indices = np.where(candidate_labels == cluster)[0]
        
        # 获取该簇的嵌入和源标签
        cluster_embedding = candidate_umap[cluster_indices]
        cluster_source_labels = [candidate_source_labels[i] for i in cluster_indices]
        
        # 为每个类别绘制点
        for category in set(candidate_source_labels):
            # 获取该簇中该类别的样本索引
            category_indices = [i for i, sl in enumerate(cluster_source_labels) if sl == category]
            
            if category_indices:
                # 获取该簇中该类别的样本嵌入
                category_embedding = cluster_embedding[category_indices]
                
                # 绘制散点图
                plt.scatter(
                    category_embedding[:, 0],
                    category_embedding[:, 1],
                    c=cand_colors[cluster],
                    marker=category_markers.get(category, 'o'),
                    label=f'Cluster {cluster} - {category_names.get(category, "Unknown")}',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5
                )
    
    plt.title('Candidate Domain Clusters', fontsize=32, fontweight='bold', pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=24, fontweight='bold',labelpad=20)
    plt.ylabel('UMAP Dimension 2', fontsize=24, fontweight='bold',labelpad=20)
    
    # 添加匹配信息
    # 检查rule_matched_clusters是否提供
    if rule_matched_clusters:
        # 构建匹配信息文本
        rule_info = []
        for rule_name, clusters in rule_matched_clusters.items():
            rule_info.append(f"{rule_name}: {len(clusters)}")
        
        rule_text = ", ".join(rule_info)
        plt.figtext(0.5, 0.01, f'Matched Pairs: {len(matching_pairs)} ({rule_text})', 
                   ha='center', fontsize=14)
    else:
        plt.figtext(0.5, 0.01, f'Matched Pairs: {len(matching_pairs)}', 
                   ha='center', fontsize=14)
    
    # 保存图像
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'matching_visualization.png'), dpi=300, bbox_inches='tight')
    output_path = os.path.join(save_dir, f'matching_visualization.{output_format}')
    save_kwargs = {'dpi': 300, 'bbox_inches': 'tight'}
    if output_format == 'tif':
        save_kwargs.update({'format': 'tiff', 'pil_kwargs': {'compression': 'none'}})
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    
    print(f"匹配可视化已保存到 '{output_path}'")
    
    # 返回训练好的UMAP模型，以便其他函数使用
    if trained_umap is None:
        return reducer
    else:
        return trained_umap        


# 2-5. 画热图（similarity矩阵）
def plot_similarity_heatmap(similarity_matrix, x_labels, y_labels, title, save_path, save_tiff=True):
    """
    绘制相似度矩阵热图
    
    参数:
    similarity_matrix: 2D numpy数组
    x_labels: X轴标签列表
    y_labels: Y轴标签列表
    title: 图像标题
    save_path: 保存路径
    """
    plt.figure(figsize=(12, 10))
    
    # 绘制热图
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=x_labels,
                yticklabels=y_labels)
    
    plt.title(title, fontsize=28, fontweight='bold', pad=20)
    plt.xlabel("Candidate", fontsize=24, fontweight='bold', labelpad = 20)
    plt.ylabel("Reference", fontsize=24, fontweight='bold', labelpad = 20)
    plt.tick_params(axis='both', which='major', labelsize=18,pad=10)
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path, dpi=300)
    if save_tiff:
        plt.savefig(save_path.replace('.png', '.tif'), dpi=300, format='tiff', pil_kwargs={'compression': 'none'})
    plt.close()
    print(f"  热图已保存至: {save_path}")

# 2-6. 画单细胞filter之后的簇umap图
def visualize_filtered_clusters(
    reference_results, 
    candidate_results, 
    matching_pairs,
    save_dir,
    reference_umap=None,
    rule_matched_clusters=None,
    target_ref_indices=None,
    thumbnail_size=30, 
    max_images=1000,
    passed_indices_dict=None,   # ← 新增参数
                                # 格式: {'rule1': [idx1, idx2, ...], 'rule2': [...]}
                                # 如果传入，直接用这些索引，不再内部筛选
):

    """
    可视化匹配的簇中保留的单细胞，保持与原始UMAP位置一致
    
    参数:
    reference_results: 参考域结果字典
    candidate_results: 候选域结果字典
    matching_pairs: 匹配对列表，每个元素为(ref_idx, cand_idx, similarity)
    save_dir: 保存目录
    reference_umap: 参考UMAP嵌入（如果有）
    rule_matched_clusters: 字典，键为规则名称（如'rule1', 'rule2', 'rule3'），值为该规则匹配的簇列表
    target_ref_indices: 目标参考簇索引列表，默认为[0, 5]
    thumbnail_size: 基础缩略图大小
    max_images: 每个簇最大显示的图像数量
    """

    print("为匹配的簇创建可视化（保持原始UMAP位置）...")
    
    # 获取候选域的特征、标签和图像路径
    features = candidate_results['features']
    labels = candidate_results['labels']
    source_labels = candidate_results['source_labels']
    image_paths = candidate_results['image_paths']
    
    # 计算细胞级别的相似度
    cell_sims = cosine_similarity(features, reference_results['cluster_centers'])
    
    # 如果未指定目标参考簇索引，则使用默认值[0, 5]
    if target_ref_indices is None:
        target_ref_indices = [0, 5]  # 默认为R0和R5
    
    # 确定需要处理的簇
    matched_clusters = []
    
    # 如果提供了rule_matched_clusters，从中提取匹配的簇
    if rule_matched_clusters:
        for rule_name, clusters in rule_matched_clusters.items():
            for cluster in clusters:
                matched_clusters.append((cluster, rule_name))
    # 否则，从matching_pairs中提取（兼容旧代码）
    elif not matched_clusters:
        # 这是旧的逻辑，为了兼容性保留
        for ref_idx, cand_idx, _ in matching_pairs:
            if ref_idx == 5:  # 对应规则1
                matched_clusters.append((cand_idx, 'rule1'))
            elif ref_idx == 0:  # 对应规则2
                matched_clusters.append((cand_idx, 'rule2'))
    
    # 获取或计算细胞大小
    if hasattr(candidate_results, 'log_sizes'):
        log_sizes = candidate_results['log_sizes']
    else:
        # 计算每个图像的细胞大小
        print("计算细胞大小...")
        cell_sizes = []
        for img_path in tqdm(image_paths, desc="计算细胞大小"):
            size = count_non_background_pixels(img_path)
            cell_sizes.append(size)
        
        # 对细胞大小进行对数变换
        log_sizes = np.log1p(np.array(cell_sizes))
    
    # 归一化对数变换后的细胞大小，用于缩放图像
    if np.max(log_sizes) > np.min(log_sizes):
        normalized_log_sizes = (log_sizes - np.min(log_sizes)) / (np.max(log_sizes) - np.min(log_sizes))
        # 将归一化值缩放到合理的图像大小范围（0.5到2倍基础大小）
        size_scale_factors = 0.5 + 1.5 * normalized_log_sizes
    else:
        size_scale_factors = np.ones_like(log_sizes)
    
    # 创建保存目录
    matched_vis_dir = os.path.join(save_dir, 'matched_clusters_visualization')
    os.makedirs(matched_vis_dir, exist_ok=True)
    
    # 为每个匹配的簇创建可视化
    for cluster_idx, rule_type in matched_clusters:
        cluster_indices = np.where(labels == cluster_idx)[0]

        # ── 筛选逻辑 ────────────────────────────────────────────────
        if passed_indices_dict is not None:
            # 【新版】直接使用外部传入的通过索引
            # 取"属于本簇" 且 "在外部通过列表里"的交集
            rule_passed = set(passed_indices_dict.get(rule_type, []))
            filtered_indices = [
                idx for idx in cluster_indices
                if idx in rule_passed
            ]
        else:
            # 【原版兼容】内部用 argmax 筛选
            filtered_indices = []
            for idx in cluster_indices:
                most_similar_ref = np.argmax(cell_sims[idx])
                if most_similar_ref in target_ref_indices:
                    filtered_indices.append(idx)
        
        if not filtered_indices:
            print(f"  簇 {cluster_idx} ({rule_type}) 没有与指定参考簇匹配的细胞，跳过")
            continue
        
        # 如果样本太多，随机采样
        if len(filtered_indices) > max_images:
            np.random.seed(42)
            selected_indices = np.random.choice(filtered_indices, max_images, replace=False)
        else:
            selected_indices = filtered_indices
        
        # 获取筛选后的特征、源标签和图像路径
        cluster_features = features[cluster_indices]  # 所有簇内细胞的特征
        filtered_features = features[filtered_indices]  # 筛选后的细胞特征
        
        # 获取所有簇内细胞的标签和路径
        cluster_source_labels = [source_labels[i] for i in cluster_indices]
        cluster_image_paths = [image_paths[i] for i in cluster_indices]
        cluster_size_factors = [size_scale_factors[i] for i in cluster_indices]
        
        # 获取筛选后细胞的标签和路径
        filtered_source_labels = [source_labels[i] for i in filtered_indices]
        filtered_image_paths = [image_paths[i] for i in filtered_indices]
        filtered_size_factors = [size_scale_factors[i] for i in filtered_indices]
        
        # 为整个簇计算UMAP嵌入，与improved_visualize_clusters_with_images一致
        if len(cluster_features) == 1:
            # 如果只有一个样本，直接将其放在原点
            cluster_embedding = np.array([[0.0, 0.0]])
        elif len(cluster_features) == 2:
            # 如果只有两个样本，将它们放在x轴上相距为1的位置
            cluster_embedding = np.array([[-0.5, 0.0], [0.5, 0.0]])
        elif len(cluster_features) <= 5:
            # 如果样本数量小于等于5，使用PCA
            try:
                from sklearn.decomposition import PCA
                cluster_embedding = PCA(n_components=2).fit_transform(cluster_features)
            except Exception as e:
                print(f"  簇 {cluster_idx} PCA降维失败: {e}，使用随机投影")
                # 如果PCA失败，使用随机投影
                cluster_embedding = np.random.randn(len(cluster_features), 2) * 0.5
        else:
            # 使用UMAP进行可视化降维
            cluster_reducer = umap.UMAP(
                n_neighbors=min(15, len(cluster_features)-1),  # 避免n_neighbors大于样本数
                min_dist=0.1,
                n_components=2,
                metric='euclidean',
                spread=5.0,    # 大幅增加扩散参数，使点更分散
                random_state=42
            )
            cluster_embedding = cluster_reducer.fit_transform(cluster_features)
        
        # 创建筛选后细胞的索引映射（从cluster_indices到filtered_indices）
        filtered_mask = np.zeros(len(cluster_indices), dtype=bool)
        for idx in filtered_indices:
            pos = np.where(cluster_indices == idx)[0]
            if len(pos) > 0:
                filtered_mask[pos[0]] = True
        
        # 设置字体
        set_plot_font()
        
        # 创建图像
        plt.figure(figsize=(16, 16))
        
        # 绘制所有簇内细胞的点（淡色）
        plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], 
                   c='lightgray', alpha=0.2, s=30)
        
        # 突出显示筛选后的细胞（亮色）
        plt.scatter(cluster_embedding[filtered_mask, 0], cluster_embedding[filtered_mask, 1], 
                   c=[f'#{category_colors[sl][0]:02x}{category_colors[sl][1]:02x}{category_colors[sl][2]:02x}' 
                     for sl in filtered_source_labels], 
                   alpha=0.8, s=80)
        
        # 计算点之间的平均距离，用于缩放图像大小
        if len(cluster_embedding) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(cluster_embedding)
            avg_distance = np.mean(distances)
            # 根据平均距离调整基础缩略图大小
            base_size = min(thumbnail_size, avg_distance * 10)
            base_size = max(base_size, 20)
        else:
            base_size = thumbnail_size
        
        # 只为筛选后的细胞显示图像
        for i, idx in enumerate(np.where(filtered_mask)[0]):
            x, y = cluster_embedding[idx, 0], cluster_embedding[idx, 1]
            source_label = cluster_source_labels[idx]
            img_path = cluster_image_paths[idx]
            size_factor = cluster_size_factors[idx]
            
            try:
                # 加载图像
                img = Image.open(img_path).convert('RGB')
                
                # 获取原始图像的宽高比
                width, height = img.size
                aspect_ratio = width / height
                
                # 根据细胞大小（对数变换后）调整图像大小，同时保持长宽比
                img_size = int(base_size * size_factor)
                
                if aspect_ratio >= 1:  # 宽图
                    new_width = img_size
                    new_height = int(img_size / aspect_ratio)
                else:  # 高图
                    new_height = img_size
                    new_width = int(img_size * aspect_ratio)
                
                # 调整图像大小，保持长宽比
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # 将PIL图像转换为numpy数组
                img_array = np.array(img)
                
                # 获取边框颜色
                border_color = category_colors.get(source_label, (0, 0, 0))
                
                # 直接在图像上添加边框
                border_width = 2
                # 上边框
                img_array[:border_width, :, :] = border_color
                # 下边框
                img_array[-border_width:, :, :] = border_color
                # 左边框
                img_array[:, :border_width, :] = border_color
                # 右边框
                img_array[:, -border_width:, :] = border_color
                
                # 将numpy数组转换回PIL图像
                bordered_img = Image.fromarray(img_array)
                
                # 创建OffsetImage
                imagebox = OffsetImage(bordered_img, zoom=1.0)
                imagebox.image.axes = plt.gca()
                
                # 创建AnnotationBbox - 不带边框
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
                plt.gca().add_artist(ab)
                
            except Exception as e:
                print(f"无法显示图像 {img_path}: {e}")
                continue
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', 
                     markerfacecolor=f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}', 
                     markersize=16, label=category_names[label])
            for label, c in category_colors.items()
            if label in filtered_source_labels
        ]
        # 添加一个表示未筛选细胞的图例元素
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='lightgray', 
                     markersize=16, label='filtered cells')
        )
        plt.legend(handles=legend_elements, 
                   loc='upper right', 
                   prop={'weight': 'bold', 'size': 16},
                   labelspacing=0.5,
                   borderpad=1.0,
                   handletextpad=0.5,
                   framealpha=0.9
        )
        
        # 获取匹配的参考簇
        matched_ref = None
        for ref_idx, cand_idx, _ in matching_pairs:
            if cand_idx == cluster_idx:
                matched_ref = ref_idx
                break
        
        # 设置标题和轴标签
        plt.title(f"{rule_type} - Cluster {cluster_idx} Matched Cells ({len(filtered_indices)}/{len(cluster_indices)} cells)", 
                 fontsize=32, fontweight='bold',pad=20)
        plt.xlabel('UMAP Dimension 1', fontsize=28, fontweight='bold',labelpad=20)
        plt.ylabel('UMAP Dimension 2', fontsize=28, fontweight='bold',labelpad=20)
        
        # 调整坐标轴范围以适应图像
        plt.axis('equal')
        
        # 自动调整坐标轴，确保所有图像都可见
        plt.margins(0.2)
        
        # 增加边距，确保图像不会被裁剪
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=18, pad=10)
        for tick in plt.gca().get_xticklabels():
            tick.set_fontweight('bold')
        for tick in plt.gca().get_yticklabels():
            tick.set_fontweight('bold')
        
        # 保存图像
        # plt.savefig(os.path.join(matched_vis_dir, f"{rule_type}_cluster{cluster_idx}_matched_cells.png"), 
        #            dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(matched_vis_dir, f"{rule_type}_cluster{cluster_idx}_matched_cells.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print("  匹配簇可视化完成")
    return matched_vis_dir
