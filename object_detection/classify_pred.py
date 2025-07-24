from typing import List, Tuple, Dict
import numpy as np
from pred_gt_matching import match_predictions_to_gt_hungarian, match_predictions_to_gt_greedy


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two sets of bounding boxes.
    Assumes box1 and box2 are in format [x_min, y_min, x_max, y_max].

    Args:
        box1 (np.array): A NumPy array of bounding boxes of shape (n, 4).
        box2 (np.array): A NumPy array of bounding boxes of shape (m, 4).

    Returns:
        np.array: A NumPy array of shape (n, m) containing the IoU scores for each pair of boxes.
    """
    # Ensure inputs are NumPy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)

    # Handle empty input arrays
    if box1.shape[0] == 0 or box2.shape[0] == 0:
        return np.zeros((box1.shape[0], box2.shape[0]))

    # Reshape box1 for broadcasting: (n, 1, 4)
    box1 = box1[:, np.newaxis, :]
    # Reshape box2 for broadcasting: (1, m, 4)
    box2 = box2[np.newaxis, :, :]

    # Determine the coordinates of the intersection rectangles
    x_min_inter = np.maximum(box1[:, :, 0], box2[:, :, 0])
    y_min_inter = np.maximum(box1[:, :, 1], box2[:, :, 1])
    x_max_inter = np.minimum(box1[:, :, 2], box2[:, :, 2])
    y_max_inter = np.minimum(box1[:, :, 3], box2[:, :, 3])

    # Calculate the area of the intersection rectangles
    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # Calculate the area of both sets of bounding boxes
    box1_area = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1])
    box2_area = (box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1])

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Handle the case of zero union area to avoid division by zero
    # Use np.where to return 0.0 where union_area is 0, and the calculated IoU otherwise.
    iou = np.where(union_area == 0, 0.0, inter_area / union_area)

    return iou



def classify_predictions_per_image(predictions: List[Dict],
                                 ground_truths: List[Dict],
                                 iou_matrix: np.ndarray,
                                 iou_threshold: float = 0.5,
                                 use_hungarian: bool = False) -> Dict:
    """
    Classify predictions as TP or FP for a single image and single class.

    Args:
        predictions: List of prediction dicts with keys: ['bbox', 'score', 'class']
        ground_truths: List of ground truth dicts with keys: ['bbox', 'class']
        iou_matrix: Shape (num_predictions, num_gt)
        iou_threshold: IoU threshold for positive matches
        use_hungarian: Whether to use Hungarian algorithm (optimal) or greedy matching

    Returns:
        Dictionary with classification results
    """
    if use_hungarian:
        matched_pred_idx, matched_gt_idx, unmatched_pred_idx = match_predictions_to_gt_hungarian(
            iou_matrix, iou_threshold
        )
    else:
        matched_pred_idx, matched_gt_idx, unmatched_pred_idx = match_predictions_to_gt_greedy(
            iou_matrix, iou_threshold
        )

    # Classify predictions
    tp_predictions = []
    fp_predictions = []

    # True positives - matched predictions
    for pred_idx in matched_pred_idx:
        pred_info = predictions[pred_idx].copy()
        pred_info['classification'] = 'TP'
        tp_predictions.append(pred_info)

    # False positives - unmatched predictions
    for pred_idx in unmatched_pred_idx:
        pred_info = predictions[pred_idx].copy()
        pred_info['classification'] = 'FP'
        fp_predictions.append(pred_info)

    # False negatives - unmatched ground truths
    all_gt_indices = set(range(len(ground_truths)))
    matched_gt_set = set(matched_gt_idx)
    unmatched_gt_indices = list(all_gt_indices - matched_gt_set)

    fn_ground_truths = []
    for gt_idx in unmatched_gt_indices:
        gt_info = ground_truths[gt_idx].copy()
        gt_info['classification'] = 'FN'
        fn_ground_truths.append(gt_info)

    return {
        'tp_predictions': tp_predictions,
        'fp_predictions': fp_predictions,
        'fn_ground_truths': fn_ground_truths,
        'num_tp': len(tp_predictions),
        'num_fp': len(fp_predictions),
        'num_fn': len(fn_ground_truths)
    }

def process_single_class_single_image(predictions: List[Dict],
                                    ground_truths: List[Dict],
                                    target_class: int,
                                    iou_matrix: np.ndarray,
                                    iou_threshold: float = 0.5,
                                    use_hungarian: bool = False) -> Dict:
    """
    Complete processing pipeline for one class in one image.

    Args:
        predictions: All predictions for the image
        ground_truths: All ground truths for the image
        target_class: Class ID to process
        iou_matrix: IoU matrix for this class only
        iou_threshold: IoU threshold for matching
        use_hungarian: Matching algorithm choice

    Returns:
        Classification results for this class in this image
    """
    # Filter predictions and ground truths for target class
    class_predictions = [p for p in predictions if p['class'] == target_class]
    class_ground_truths = [gt for gt in ground_truths if gt['class'] == target_class]

    if len(class_predictions) == 0 and len(class_ground_truths) == 0:
        return {
            'tp_predictions': [],
            'fp_predictions': [],
            'fn_ground_truths': [],
            'num_tp': 0,
            'num_fp': 0,
            'num_fn': 0
        }
    elif len(class_predictions) == 0:
        # No predictions, all ground truths are FN
        fn_ground_truths = [dict(gt, classification='FN') for gt in class_ground_truths]
        return {
            'tp_predictions': [],
            'fp_predictions': [],
            'fn_ground_truths': fn_ground_truths,
            'num_tp': 0,
            'num_fp': 0,
            'num_fn': len(fn_ground_truths)
        }
    elif len(class_ground_truths) == 0:
        # No ground truths, all predictions are FP
        fp_predictions = [dict(p, classification='FP') for p in class_predictions]
        return {
            'tp_predictions': [],
            'fp_predictions': fp_predictions,
            'fn_ground_truths': [],
            'num_tp': 0,
            'num_fp': len(fp_predictions),
            'num_fn': 0
        }

    # Normal case - both predictions and ground truths exist
    return classify_predictions_per_image(
        class_predictions, class_ground_truths, iou_matrix,
        iou_threshold, use_hungarian
    )

# Example usage
def example_usage():
    """
    Example of how to use the matching functions.
    """
    # Example predictions (should be sorted by confidence score descending)
    predictions = [
        {'bbox': [10, 10, 50, 50], 'score': 0.9, 'class': 1},
        {'bbox': [60, 60, 100, 100], 'score': 0.8, 'class': 1},
        {'bbox': [15, 15, 45, 45], 'score': 0.7, 'class': 1},  # Overlaps with first
    ]

    # Example ground truths
    ground_truths = [
        {'bbox': [12, 12, 48, 48], 'class': 1},  # Should match first prediction
        {'bbox': [200, 200, 240, 240], 'class': 1},  # No matching prediction
    ]

    # Example IoU matrix (you would calculate this with your IoU function)
    # Shape: (num_predictions, num_ground_truths)
    iou_matrix = np.array([
        [0.85, 0.0],   # First prediction: high IoU with first GT, no overlap with second
        [0.1, 0.0],    # Second prediction: low IoU with both GTs
        [0.75, 0.0],   # Third prediction: medium-high IoU with first GT
    ])

    # Process this class for this image
    results = process_single_class_single_image(
        predictions, ground_truths, target_class=1,
        iou_matrix=iou_matrix, iou_threshold=0.5,
        use_hungarian=True
    )

    print(f"TP: {results['num_tp']}")
    print(f"FP: {results['num_fp']}")
    print(f"FN: {results['num_fn']}")

    return results

if __name__ == "__main__":
    

    # IoU Example usage with arrays:
    boxes_a = np.array([[10, 10, 50, 50], [20, 20, 60, 60]])
    boxes_b = np.array([[15, 15, 55, 55], [30, 30, 70, 70]])
    iou_matrix = calculate_iou(boxes_a, boxes_b)
    print(f"IoU Matrix:\n{iou_matrix}")
    
    example_usage()


