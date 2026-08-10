import sys, os
sys.path.insert(0, r'E:\limr\Soft\shilab-cluster-algorithm\cross_cluster_matching\application')
from pipeline import run_pipeline
run_pipeline(
    positive_folder_path        = r'E:/lyy_test/260722_time_test/binary_infer',
    negative_folder_path        = r'J:/limr/effusion/result/binary_infer/d612/densenet161/nothing',
    malignant_cells_dir         = r'J:/limr/effusion/data/classification_model/effusion_singlecell_5.0/malignant',
    benign_cells_dir            = r'J:/limr/effusion/data/classification_model/effusion_singlecell_5.0/benign',
    model_path                  = r'J:/limr/effusion/result/effusion_classification_sc_out_5.0/train_val_models/DenseNet161/densenet161_fold_4.pth',
    base_save_dir               = r'E:/lyy_test/260722_time_test/clustering',
    model_type                  = 'DenseNet161',
    feature_layer               = 'penultimate',
    reference_k                 = 10,
    max_candidate_k             = 20,
    pca_dim                     = 32,
    batch_size                  = 16,
    num_workers                 = 0,
    cell_size_weight            = 1.0,
    mean                        = [0.5665, 0.7001, 0.765],
    std                         = [0.3161, 0.2253, 0.1554],
    malignant_ref_clusters      = [7, 0],
    matching_rules              = [{'target': 7, 'avoid': [0], 'threshold_target': 0.9, 'threshold_ratio': 3.0, 'threshold_avoid_others': 0.1}, {'target': 0, 'avoid': [7], 'threshold_target': 0.6, 'threshold_ratio': 3.0, 'threshold_avoid_others': 0.2}],
    use_cell_level_rule_filter  = True,
)