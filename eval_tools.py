import numpy as np
from scipy.optimize import linear_sum_assignment


def angular_difference(a,b):
    x = np.abs(a-b)
    return np.min(np.array((x, np.abs(360-x))), axis=0)


def compute_metrics(predictions, groundtruth, threshold):
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
        "f1_score": f1,
        "accuracy": accuracy,
        "mean_error": mean_error
    }


if __name__ == "__main__":
    
    from pprint import pprint
    
    # Example usage:
    predictions = [10, 80, 200]     # Example predicted DOAs in degrees
    groundtruth = [15, 85]          # Example groundtruth DOAs in degrees
    threshold = 300                  # Example threshold in degrees

    metrics = compute_metrics(predictions, groundtruth, threshold)
    pprint(metrics)