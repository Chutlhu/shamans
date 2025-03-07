import numpy as np
from scipy.optimize import linear_sum_assignment


def angular_difference(a,b):
    x = np.abs(a-b)
    return np.min(np.array((x, np.abs(360-x))), axis=0)


def compute_metrics(predictions, groundtruth, threshold, candidate_doas):
    # Build the cost matrix: rows for predictions, columns for groundtruth.
    cost_matrix = np.zeros((len(predictions), len(groundtruth)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(groundtruth):
            cost_matrix[i, j] = angular_difference(pred, gt)
    
    # Solve the assignment problem using the Hungarian algorithm.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter assignments based on the threshold.
    true_positive_pairs = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < threshold:
            true_positive_pairs.append((i, j))
    
    # Count the metrics.    
    TP = len(true_positive_pairs)
    FP = len(predictions) - TP
    FN = len(groundtruth) - TP
    
    TN = compute_true_negatives_with_candidates(groundtruth, threshold, candidate_doas)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0    
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    
    # Mean error for true positive pairs.
    errors = [cost_matrix[i, j] for i, j in true_positive_pairs]
    mean_error = np.mean(errors) if errors else None
    
    return {
        "true_positives": TP,
        "false_positives": FP,
        "false_negatives": FN,
        "precision": precision,
        "recall": recall,
        "tpr": TPR,
        "fpr": FPR,
        "f1_score": f1,
        "accuracy": accuracy,
        "mean_error": mean_error
    }
    
def compute_true_negatives_with_candidates(groundtruth, threshold, candidate_doas):
    """
    Computes FPR using a provided list of candidate DOAs.
    
    For each candidate DOA, if it is not within the threshold of any groundtruth, 
    it is counted as a negative (TN). FP is computed from the detection metric.
    """
    # Count true negatives: candidate DOAs that are not 'covered' by any groundtruth.
    TN = 0
    for candidate in candidate_doas:
        # A candidate is covered if it lies within the threshold of any groundtruth.
        is_covered = any(angular_difference(candidate, gt) <= threshold for gt in groundtruth)
        if not is_covered:
            TN += 1
    return TN

# def compute_true_negatives(groundtruth, threshold, grid_resolution=6):
#     """
#     Define the negatives by discretizing 360 degrees. Each bin not covered by a groundtruth 
#     (within the threshold) is considered a negative.
#     """
#     # Total bins in 360 degrees
#     total_bins = int(360 / grid_resolution)
    
#     # Mark bins covered by groundtruth (extend each groundtruth by threshold on either side)
#     gt_coverage = np.zeros(total_bins, dtype=bool)
#     for gt in groundtruth:
#         # Determine bins within threshold of the groundtruth source
#         start = int((gt - threshold) % 360 / grid_resolution)
#         end = int((gt + threshold) % 360 / grid_resolution)
#         if start <= end:
#             gt_coverage[start:end+1] = True
#         else:
#             # Wrap-around case
#             gt_coverage[:end+1] = True
#             gt_coverage[start:] = True

#     TN = np.sum(~gt_coverage)  # bins that are not "groundtruth-covered"
    
#     # # FP remains the same from our detector (extra predictions that were not matched)
#     # _, FP, _, _ = compute_detection_metrics(predictions, groundtruth, threshold)
    
#     # FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
#     return TN


if __name__ == "__main__":
    
    from pprint import pprint
    
    # Example usage:
    predictions = [10, 80, 200]     # Example predicted DOAs in degrees
    groundtruth = [15, 85]          # Example groundtruth DOAs in degrees
    threshold = 300                  # Example threshold in degrees

    metrics = compute_metrics(predictions, groundtruth, threshold)
    pprint(metrics)