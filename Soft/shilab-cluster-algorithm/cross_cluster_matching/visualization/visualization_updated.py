def visualize_matching_clusters(reference_results, candidate_results, matching_pairs, save_dir, 
                               trained_umap=None, rule_matched_clusters=None):
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
    plt.savefig(os.path.join(save_dir, 'matching_visualization.tif'), dpi=300, bbox_inches='tight', format='tiff', pil_kwargs={'compression': 'none'})
    plt.close()
    
    print(f"匹配可视化已保存到 '{os.path.join(save_dir, 'matching_visualization.tif')}'")
    
    # 返回训练好的UMAP模型，以便其他函数使用
    if trained_umap is None:
        return reducer
    else:
        return trained_umap