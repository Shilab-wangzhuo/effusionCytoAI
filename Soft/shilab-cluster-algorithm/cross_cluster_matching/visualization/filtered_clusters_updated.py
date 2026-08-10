def visualize_filtered_clusters(
    reference_results, 
    candidate_results, 
    matching_pairs,
    save_dir,
    reference_umap=None,
    rule_matched_clusters=None,  # 改为字典格式，键为规则名称，值为该规则匹配的簇列表
    target_ref_indices=None,     # 新增参数，指定目标参考簇索引
    thumbnail_size=30, 
    max_images=1000
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
        # 获取该簇的样本索引
        cluster_indices = np.where(labels == cluster_idx)[0]
        
        # 筛选出最相似的是目标参考簇的细胞
        filtered_indices = []
        
        for idx in cluster_indices:
            # 找出最相似的参考簇
            most_similar_ref = np.argmax(cell_sims[idx])
            # 只保留最相似的是目标参考簇的细胞
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
        plt.savefig(os.path.join(matched_vis_dir, f"{rule_type}_cluster{cluster_idx}_matched_cells.tif"), 
                   dpi=150, bbox_inches='tight', format='tiff', pil_kwargs={'compression': 'none'})
        plt.close()
    
    print("  匹配簇可视化完成")
    return matched_vis_dir