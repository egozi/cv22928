import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Any, Tuple

class ObjectDetectionEvaluator:
    def __init__(self, class_names: List[str]):
        """
        Initialize the evaluator with class names.

        Args:
            class_names: List of class names (index corresponds to class_id)
        """
        self.class_names = class_names
        self.results_storage = defaultdict(list)  # {class_id: [results_per_image]}

    def calculate_iou_matrix_single_class(self, predictions: List[Dict],
                                        ground_truths: List[Dict],
                                        class_id: int) -> np.ndarray:
        """
        Calculate IoU matrix between predictions and ground truths for a specific class.

        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries
            class_id: Target class ID to filter by

        Returns:
            IoU matrix of shape (num_preds_class, num_gts_class)
        """
        # Filter predictions and ground truths by class
        class_preds = [p for p in predictions if p.get('class', p.get('class_id')) == class_id]
        class_gts = [g for g in ground_truths if g.get('class', g.get('class_id')) == class_id]

        if len(class_preds) == 0 or len(class_gts) == 0:
            return np.array([]).reshape(len(class_preds), len(class_gts))

        iou_matrix = np.zeros((len(class_preds), len(class_gts)))

        for i, pred in enumerate(class_preds):
            for j, gt in enumerate(class_gts):
                iou_matrix[i, j] = self.calculate_iou(pred['bbox'], gt['bbox'])

        return iou_matrix

    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1, bbox2: Bounding boxes in format [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # Calculate intersection area
        if x2_i <= x1_i or y2_i <= y1_i:
            intersection = 0.0
        else:
            intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def store_results(self, image_id: str, class_id: int, results: Dict[str, Any]):
        """
        Store results for a single image and class.

        Args:
            image_id: Image identifier
            class_id: Class identifier
            results: Results dictionary from process_single_class_single_image
        """
        # Add image_id to results for tracking
        results_with_id = results.copy()
        results_with_id['image_id'] = image_id

        self.results_storage[class_id].append(results_with_id)

    def aggregate_results_for_class(self, class_id: int) -> List[Dict[str, Any]]:
        """
        Aggregate all predictions for a specific class across all images.

        Args:
            class_id: Class identifier

        Returns:
            List of all predictions with their classifications and scores
        """
        all_predictions = []

        for image_results in self.results_storage[class_id]:
            # Add TP predictions
            for pred in image_results['tp_predictions']:
                all_predictions.append({
                    'score': pred['score'],
                    'classification': 'TP',
                    'image_id': image_results['image_id'],
                    'bbox': pred['bbox']
                })

            # Add FP predictions
            for pred in image_results['fp_predictions']:
                all_predictions.append({
                    'score': pred['score'],
                    'classification': 'FP',
                    'image_id': image_results['image_id'],
                    'bbox': pred['bbox']
                })

        # Sort by confidence score in descending order
        all_predictions.sort(key=lambda x: x['score'], reverse=True)

        return all_predictions

    def calculate_total_ground_truths(self, class_id: int) -> int:
        """
        Calculate total number of ground truth objects for a class.

        Args:
            class_id: Class identifier

        Returns:
            Total number of ground truth objects
        """
        total_gts = 0
        for image_results in self.results_storage[class_id]:
            total_gts += image_results['num_tp'] + image_results['num_fn']

        return total_gts

    def calculate_precision_recall_curve(self, class_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate precision-recall curve for a specific class.

        Args:
            class_id: Class identifier

        Returns:
            Tuple of (precision_values, recall_values, thresholds)
        """
        predictions = self.aggregate_results_for_class(class_id)
        total_gts = self.calculate_total_ground_truths(class_id)

        if len(predictions) == 0 or total_gts == 0:
            return np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([])

        # Initialize arrays
        precisions = []
        recalls = []
        thresholds = []

        tp_count = 0
        fp_count = 0

        # Calculate precision and recall at each threshold
        for i, pred in enumerate(predictions):
            if pred['classification'] == 'TP':
                tp_count += 1
            else:  # FP
                fp_count += 1

            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / total_gts if total_gts > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            thresholds.append(pred['score'])

        # Add point (0, 1) at the beginning for complete curve
        precisions = [1.0] + precisions
        recalls = [0.0] + recalls

        return np.array(precisions), np.array(recalls), np.array(thresholds)

    def calculate_average_precision(self, class_id: int, interpolation='11point') -> float:
        """
        Calculate Average Precision (AP) for a specific class.

        Args:
            class_id: Class identifier
            interpolation: '11point' for 11-point interpolation or 'all' for all points

        Returns:
            Average Precision value
        """
        precisions, recalls, _ = self.calculate_precision_recall_curve(class_id)

        if len(precisions) <= 1:
            return 0.0

        if interpolation == '11point':
            # 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                # Find precisions where recall >= t
                valid_precisions = precisions[recalls >= t]
                max_precision = np.max(valid_precisions) if len(valid_precisions) > 0 else 0.0
                ap += max_precision
            return ap / 11.0

        else:  # All point interpolation (more accurate)
            # Sort by recall
            sorted_indices = np.argsort(recalls)
            sorted_recalls = recalls[sorted_indices]
            sorted_precisions = precisions[sorted_indices]

            # Remove duplicate recall values, keeping the maximum precision
            unique_recalls = []
            max_precisions = []

            i = 0
            while i < len(sorted_recalls):
                current_recall = sorted_recalls[i]
                max_prec = sorted_precisions[i]

                # Find all entries with the same recall
                while i < len(sorted_recalls) and sorted_recalls[i] == current_recall:
                    max_prec = max(max_prec, sorted_precisions[i])
                    i += 1

                unique_recalls.append(current_recall)
                max_precisions.append(max_prec)

            # Calculate area under curve
            ap = 0.0
            for i in range(1, len(unique_recalls)):
                ap += (unique_recalls[i] - unique_recalls[i-1]) * max_precisions[i]

            return ap

    def plot_precision_recall_curves(self, figsize=(6, 4), save_path=None):
        """
        Plot precision-recall curves for all classes.

        Args:
            figsize: Figure size tuple
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=figsize)

        # colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        colors = [
            'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
            'lime', 'deeppink', 'gold', 'aqua', 'orangered', 'dodgerblue', 'crimson'
        ]

        for class_id, class_name in enumerate(self.class_names):
            if class_id not in self.results_storage:
                continue

            precisions, recalls, _ = self.calculate_precision_recall_curve(class_id)
            ap = self.calculate_average_precision(class_id, interpolation='all')

            plt.plot(recalls, precisions,
                    color=colors[class_id],
                    linewidth=2,
                    label=f'{class_name} (AP={ap:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def calculate_mean_average_precision(self, interpolation='all') -> float:
        """
        Calculate mean Average Precision (mAP) across all classes.

        Args:
            interpolation: '11point' for 11-point interpolation or 'all' for all points

        Returns:
            Mean Average Precision value
        """
        aps = []
        for class_id in range(len(self.class_names)):
            if class_id in self.results_storage:
                ap = self.calculate_average_precision(class_id, interpolation)
                aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def print_evaluation_summary(self):
        """
        Print a summary of evaluation results.
        """
        print("Object Detection Evaluation Summary")
        print("=" * 50)

        total_images = len(set(result['image_id']
                              for class_results in self.results_storage.values()
                              for result in class_results))
        print(f"Total images evaluated: {total_images}")
        print(f"Total classes: {len(self.class_names)}")
        print()

        # Per-class results
        for class_id, class_name in enumerate(self.class_names):
            if class_id not in self.results_storage:
                continue

            ap = self.calculate_average_precision(class_id, interpolation='all')
            total_gts = self.calculate_total_ground_truths(class_id)
            predictions = self.aggregate_results_for_class(class_id)
            total_tps = sum(1 for p in predictions if p['classification'] == 'TP')
            total_fps = sum(1 for p in predictions if p['classification'] == 'FP')
            totlal_fns = total_gts - total_tps

            print(f"Class: {class_name}")
            print(f"  Average Precision (AP): {ap:.3f}")
            print(f"  Ground Truth objects: {total_gts}")
            print(f"  True Positives: {total_tps}")
            print(f"  False Positives: {total_fps}")
            print(f"  False Negatives: {totlal_fns}")
            print()

        # Overall mAP
        map_score = self.calculate_mean_average_precision(interpolation='all')
        print(f"Mean Average Precision (mAP): {map_score:.3f}")



