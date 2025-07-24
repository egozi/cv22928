import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment


def match_predictions_to_gt_hungarian(iou_matrix: np.ndarray, iou_threshold: float = 0.5) -> Tuple[List[int], List[int], List[int]]:
    """
    Match predictions to ground truth using Hungarian algorithm (optimal assignment).

    Args:
        iou_matrix: Shape (num_predictions, num_gt). IoU between each pred and gt box
        iou_threshold: Minimum IoU to consider a match valid

    Returns:
        matched_pred_indices: List of prediction indices that were matched
        matched_gt_indices: List of corresponding ground truth indices
        unmatched_pred_indices: List of prediction indices that couldn't be matched
    """
    if iou_matrix.size == 0:
        return [], [], list(range(iou_matrix.shape[0])) if iou_matrix.shape[0] > 0 else []

    # Hungarian algorithm works with cost matrix (we want to maximize IoU, so use negative)
    cost_matrix = -iou_matrix

    # Find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    matched_pred_indices = []
    matched_gt_indices = []

    # Filter matches based on IoU threshold
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
            matched_pred_indices.append(pred_idx)
            matched_gt_indices.append(gt_idx)

    # Find unmatched predictions
    all_pred_indices = set(range(iou_matrix.shape[0]))
    matched_pred_set = set(matched_pred_indices)
    unmatched_pred_indices = list(all_pred_indices - matched_pred_set)

    return matched_pred_indices, matched_gt_indices, unmatched_pred_indices


def match_predictions_to_gt_greedy(iou_matrix: np.ndarray, iou_threshold: float = 0.5) -> Tuple[List[int], List[int], List[int]]:
    """
    Match predictions to ground truth using greedy algorithm (faster, assumes predictions are sorted by confidence).

    Args:
        iou_matrix: Shape (num_predictions, num_gt). IoU between each pred and gt box
        iou_threshold: Minimum IoU to consider a match valid

    Returns:
        matched_pred_indices: List of prediction indices that were matched
        matched_gt_indices: List of corresponding ground truth indices
        unmatched_pred_indices: List of prediction indices that couldn't be matched
    """
    if iou_matrix.size == 0:
        return [], [], list(range(iou_matrix.shape[0])) if iou_matrix.shape[0] > 0 else []

    num_predictions, num_gt = iou_matrix.shape
    gt_matched = np.zeros(num_gt, dtype=bool)

    matched_pred_indices = []
    matched_gt_indices = []

    # Iterate through predictions (should be sorted by confidence descending)
    for pred_idx in range(num_predictions):
        best_iou = 0
        best_gt_idx = -1

        # Find best unmatched ground truth for this prediction
        for gt_idx in range(num_gt):
            if not gt_matched[gt_idx] and iou_matrix[pred_idx, gt_idx] > best_iou:
                best_iou = iou_matrix[pred_idx, gt_idx]
                best_gt_idx = gt_idx

        # If best IoU meets threshold, make the match
        if best_iou >= iou_threshold:
            matched_pred_indices.append(pred_idx)
            matched_gt_indices.append(best_gt_idx)
            gt_matched[best_gt_idx] = True

    # Find unmatched predictions
    all_pred_indices = set(range(num_predictions))
    matched_pred_set = set(matched_pred_indices)
    unmatched_pred_indices = list(all_pred_indices - matched_pred_set)

    return matched_pred_indices, matched_gt_indices, unmatched_pred_indices

