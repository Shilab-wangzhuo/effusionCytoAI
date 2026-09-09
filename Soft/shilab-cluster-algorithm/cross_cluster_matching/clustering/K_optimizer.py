# clustering/K_optimizer.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def plot_k_optimization_curve(k_values, scores, best_k, patient_id, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, 'o-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('k (Number of Clusters)', fontsize=24, fontweight='bold', labelpad=20)
    plt.ylabel('Weighted Score', fontsize=24, fontweight='bold', labelpad=20)
    plt.title(f'{patient_id} K-Optimization', fontsize=28, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16)
    plt.margins(0.1)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    save_path = os.path.join(save_dir, 'k_optimization_plot.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def optimize_k_for_candidate_clustering(
    patient_candidate_normalized,
    patient_candidate_source_labels,
    patient_candidate_image_paths,
    reference_results,
    patient_save_dir,
    patient_id,
    max_k=20,
    malignant_ref_clusters=None
):
    """
    Optimize the number of clusters k for candidate domain clustering.

    Args:
        patient_candidate_normalized  - Normalized candidate domain features
        patient_candidate_source_labels - Source labels of candidate samples
        patient_candidate_image_paths - Image paths of candidate samples
        reference_results             - Reference domain clustering result dict
        patient_save_dir              - Directory to save results
        patient_id                    - Slide ID
        max_k                         - Maximum k value to try
        malignant_ref_clusters        - Indices of malignant reference clusters

    Returns:
        best_candidate_results - Clustering results under the best k
        best_k                 - Best k value
    """
    print(f"  Candidate cell count: {len(patient_candidate_normalized)}, starting k optimization...")

    k_optimization_dir = os.path.join(patient_save_dir, 'k_optimization')
    os.makedirs(k_optimization_dir, exist_ok=True)

    k_scores = []

    # Iterate k from 2 to max_k (capped at 1/3 of sample size) with step 2
    max_k = min(max_k, len(patient_candidate_normalized) // 3)
    k_values = range(2, max_k + 1, 2)

    benign_ref_clusters = [i for i in range(reference_results['n_clusters']) if i not in malignant_ref_clusters]

    reference_centers = reference_results['cluster_centers']
    cell_similarity_matrix = cosine_similarity(patient_candidate_normalized, reference_centers)
    cell_nearest_ref_cluster = np.argmin(1 - cell_similarity_matrix, axis=1)

    for k in k_values:
        print(f"  Trying k={k}")

        try:
            k_save_dir = os.path.join(k_optimization_dir, f'k{k}')
            os.makedirs(k_save_dir, exist_ok=True)

            patient_candidate_kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300)
            patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)

            cluster_counts = np.bincount(patient_candidate_labels, minlength=k)
            if np.any(cluster_counts == 0):
                print(f"  Warning: empty cluster detected at k={k}, skipping")
                continue

            patient_candidate_distances = np.linalg.norm(
                patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels],
                axis=1
            )

            candidate_centers = patient_candidate_kmeans.cluster_centers_
            similarity_matrix = cosine_similarity(candidate_centers, reference_centers)
            distance_matrix = 1 - similarity_matrix
            cluster_nearest_ref = np.argmin(distance_matrix, axis=1)

            cluster_scores = []
            total_cells = len(patient_candidate_normalized)

            for cand_idx in range(k):
                cluster_mask = (patient_candidate_labels == cand_idx)
                cluster_size = np.sum(cluster_mask)

                # a. Distance to the nearest malignant reference cluster
                Dm = min(distance_matrix[cand_idx, ref_idx] for ref_idx in malignant_ref_clusters)

                # b. Distance to the nearest benign reference cluster
                Db = min((distance_matrix[cand_idx, ref_idx] for ref_idx in benign_ref_clusters), default=1.0)

                # c. Normalized separation score
                epsilon = 1e-10
                Ssep = abs((Dm - Db) / (Dm + Db + epsilon))

                # d. Cell-level consistency: fraction of cells whose nearest ref cluster
                #    matches the cluster center's nearest ref cluster
                cluster_cells_nearest_ref = cell_nearest_ref_cluster[cluster_mask]
                cluster_nearest_ref_cluster = cluster_nearest_ref[cand_idx]
                matching_cells_count = np.sum(cluster_cells_nearest_ref == cluster_nearest_ref_cluster)
                P_C = matching_cells_count / total_cells

                cluster_scores.append((Ssep, cluster_size, P_C))

            # e. P(C)-weighted average of separation scores
            total_P_C = sum(P_C for _, _, P_C in cluster_scores)
            weighted_score = (
                sum(Ssep * P_C for Ssep, _, P_C in cluster_scores) / total_P_C
                if total_P_C > 0 else 0
            )

            print(f"  k={k} weighted score: {weighted_score:.4f}")
            k_scores.append((k, weighted_score))

            np.save(os.path.join(k_save_dir, 'candidate_cluster_centers.npy'), candidate_centers)
            np.save(os.path.join(k_save_dir, 'candidate_labels.npy'), patient_candidate_labels)
            np.save(os.path.join(k_save_dir, 'cluster_scores.npy'), np.array(cluster_scores))

        except Exception as e:
            print(f"  Error at k={k}: {str(e)}, skipping")
            continue

    if not k_scores:
        print("  Warning: all k values failed, falling back to k=2")
        try:
            k = 2
            k_save_dir = os.path.join(k_optimization_dir, f'k{k}')
            os.makedirs(k_save_dir, exist_ok=True)

            patient_candidate_kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300)
            patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)

            patient_candidate_distances = np.linalg.norm(
                patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels],
                axis=1
            )

            best_candidate_results = {
                'features': patient_candidate_normalized,
                'labels': patient_candidate_labels,
                'distances': patient_candidate_distances,
                'source_labels': patient_candidate_source_labels,
                'image_paths': patient_candidate_image_paths,
                'kmeans': patient_candidate_kmeans,
                'cluster_centers': patient_candidate_kmeans.cluster_centers_,
                'n_clusters': k
            }

            best_k = k
            print(f"  Clustering completed with fallback k={k}")

        except Exception as e:
            print(f"  Fallback k=2 also failed: {str(e)}, skipping this slide")
            return None, None

    else:
        k_scores.sort(key=lambda x: x[1], reverse=True)
        best_k, best_score = k_scores[0]

        k_df = pd.DataFrame(k_scores, columns=['k', 'weighted_score'])
        k_df.to_csv(os.path.join(k_optimization_dir, 'k_optimization_results.csv'), index=False)

        k_vals = [x[0] for x in k_scores]
        scores = [x[1] for x in k_scores]
        plot_k_optimization_curve(k_vals, scores, best_k, patient_id, k_optimization_dir)

        patient_candidate_kmeans = KMeans(n_clusters=best_k, random_state=42, max_iter=300)
        patient_candidate_labels = patient_candidate_kmeans.fit_predict(patient_candidate_normalized)

        patient_candidate_distances = np.linalg.norm(
            patient_candidate_normalized - patient_candidate_kmeans.cluster_centers_[patient_candidate_labels],
            axis=1
        )

        best_candidate_results = {
            'features': patient_candidate_normalized,
            'labels': patient_candidate_labels,
            'distances': patient_candidate_distances,
            'source_labels': patient_candidate_source_labels,
            'image_paths': patient_candidate_image_paths,
            'kmeans': patient_candidate_kmeans,
            'cluster_centers': patient_candidate_kmeans.cluster_centers_,
            'n_clusters': best_k
        }

        print(f"  Optimization complete. Best k={best_k}, best weighted score: {best_score:.4f}")

    return best_candidate_results, best_k