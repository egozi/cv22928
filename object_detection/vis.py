from typing import Union, Dict, Optional, Tuple, List
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
# import matplotlib.patches as patches
from matplotlib import patches, patheffects
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image



def visualize_detection_results(image: Union[str, np.ndarray, Image.Image],
                                results: Dict,
                                ground_truths: Optional[List[Dict]] = None,
                                class_name: str = "Object",
                                iou_threshold: float = 0.5,
                                show_confidence: bool = True,
                                bbox_format: str = 'xyxy',
                                figsize: Tuple[int, int] = (10, 7),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize object detection results and ground truths on an image.

    Args:
        image: Image path, numpy array, or PIL Image
        results: Output from process_single_class_single_image()
        ground_truths: List of all GT boxes, each a dict with at least 'bbox'
        class_name: Display class name
        iou_threshold: Threshold used for matching
        show_confidence: Whether to show confidence score
        show_iou: Not used here
        bbox_format: 'xyxy' or 'xywh'
        figsize: Figure size
        save_path: Optional output path

    Returns:
        matplotlib Figure
    """
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found: {image}")
        img = Image.open(image)
        img_array = np.array(img)
    elif isinstance(image, Image.Image):
        img_array = np.array(image)
    elif isinstance(image, np.ndarray):
        img_array = image
    else:
        raise ValueError("Unsupported image type")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_array)
    ax.set_title(f'{class_name} Detection (IoU ≥ {iou_threshold}) | '
                 f'TP: {results["num_tp"]}, FP: {results["num_fp"]}, FN: {results["num_fn"]}',
                 fontsize=14, fontweight='bold')

    colors = {
        'TP': '#00FF00',   # green
        'FP': '#FF0000',   # red
        'FN': '#0000FF',   # blue
        'GT': '#FFA500'    # orange
    }

    def to_xyxy(bbox):
        if bbox_format == 'xyxy':
            return bbox
        elif bbox_format == 'xywh':
            x, y, w, h = bbox
            return [x, y, x + w, y + h]
        else:
            raise ValueError("Invalid bbox_format")

    # Draw ground truths
    if ground_truths:
        for gt in ground_truths:
            x1, y1, x2, y2 = to_xyxy(gt['bbox'])
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=1.5, edgecolor=colors['GT'],
                             facecolor='none', linestyle='-')
            ax.add_patch(rect)
            ax.text(x1, y2 + 5, "GT",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['GT'], alpha=0.5),
                    fontsize=9, fontweight='bold', color='black')

    # TP
    for pred in results['tp_predictions']:
        x1, y1, x2, y2 = to_xyxy(pred['bbox'])
        label = "TP"
        if show_confidence and 'score' in pred:
            label += f" ({pred['score']:.2f})"
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=3, edgecolor=colors['TP'], facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['TP'], alpha=0.7),
                fontsize=10, fontweight='bold')

    # FP
    for pred in results['fp_predictions']:
        x1, y1, x2, y2 = to_xyxy(pred['bbox'])
        label = "FP"
        if show_confidence and 'score' in pred:
            label += f" ({pred['score']:.2f})"
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=3, edgecolor=colors['FP'], facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['FP'], alpha=0.7),
                fontsize=10, fontweight='bold', color='white')

    # FN
    for gt in results['fn_ground_truths']:
        x1, y1, x2, y2 = to_xyxy(gt['bbox'])
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=3, edgecolor=colors['FN'], facecolor='none', linestyle=':')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, "FN",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['FN'], alpha=0.7),
                fontsize=10, fontweight='bold', color='white')

    # Legend
    ax.legend(handles=[
        Patch(color=colors['TP'], label='True Positive'),
        Patch(color=colors['FP'], label='False Positive'),
        Patch(color=colors['FN'], label='False Negative'),
        Patch(color=colors['GT'], label='Ground Truth')
    ], loc='upper right', fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def visualize_matching_comparison(image: Union[str, np.ndarray, Image.Image],
                                results_hungarian: Dict,
                                results_greedy: Dict,
                                class_name: str = "Object",
                                iou_threshold: float = 0.5,
                                bbox_format: str = 'xyxy',
                                figsize: Tuple[int, int] = (16, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare Hungarian vs Greedy matching results side by side.

    Args:
        image: Image to visualize
        results_hungarian: Results from Hungarian algorithm
        results_greedy: Results from Greedy algorithm
        class_name: Class name for display
        iou_threshold: IoU threshold used
        bbox_format: Bounding box format
        figsize: Figure size
        save_path: Path to save comparison

    Returns:
        matplotlib Figure object
    """

    # Load image
    if isinstance(image, str):
        img = Image.open(image)
        img_array = np.array(img)
    elif isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Colors
    colors = {'TP': '#00FF00', 'FP': '#FF0000', 'FN': '#0000FF'}

    def plot_results(ax, results, title):
        ax.imshow(img_array)
        ax.set_title(f'{title}\nTP: {results["num_tp"]} | FP: {results["num_fp"]} | FN: {results["num_fn"]}',
                    fontsize=12, fontweight='bold')

        # Helper function
        def convert_bbox(bbox):
            if bbox_format == 'xyxy':
                return bbox
            else:  # xywh
                x, y, w, h = bbox
                return [x, y, x + w, y + h]

        # Draw all classifications
        for classification, items_key in [('TP', 'tp_predictions'), ('FP', 'fp_predictions')]:
            for item in results[items_key]:
                bbox = convert_bbox(item['bbox'])
                x1, y1, x2, y2 = bbox

                style = '-' if classification == 'TP' else '--'
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor=colors[classification],
                               facecolor='none', linestyle=style)
                ax.add_patch(rect)

                label = f"{classification}"
                if 'score' in item:
                    label += f" ({item['score']:.2f})"

                text_color = 'black' if classification == 'TP' else 'white'
                ax.text(x1, y1-3, label,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[classification], alpha=0.7),
                       fontsize=8, fontweight='bold', color=text_color)

        # Draw FN
        for item in results['fn_ground_truths']:
            bbox = convert_bbox(item['bbox'])
            x1, y1, x2, y2 = bbox

            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor=colors['FN'],
                           facecolor='none', linestyle=':')
            ax.add_patch(rect)

            ax.text(x1, y1-3, "FN",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['FN'], alpha=0.7),
                   fontsize=8, fontweight='bold', color='white')

        ax.set_xticks([])
        ax.set_yticks([])

    # Plot both results
    plot_results(ax1, results_hungarian, f'{class_name} - Hungarian Matching')
    plot_results(ax2, results_greedy, f'{class_name} - Greedy Matching')

    # Add shared legend
    legend_elements = [
        patches.Patch(color=colors['TP'], label='True Positive'),
        patches.Patch(color=colors['FP'], label='False Positive'),
        patches.Patch(color=colors['FN'], label='False Negative')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")

    return fig

def create_detection_summary_plot(results_per_class: Dict[str, Dict],
                                class_names: List[str],
                                figsize: Tuple[int, int] = (12, 6),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a summary bar plot showing TP/FP/FN counts across classes.

    Args:
        results_per_class: Dict with class names as keys and results dicts as values
        class_names: List of class names for ordering
        figsize: Figure size
        save_path: Path to save the plot

    Returns:
        matplotlib Figure object
    """

    # Prepare data
    tp_counts = [results_per_class.get(cls, {'num_tp': 0})['num_tp'] for cls in class_names]
    fp_counts = [results_per_class.get(cls, {'num_fp': 0})['num_fp'] for cls in class_names]
    fn_counts = [results_per_class.get(cls, {'num_fn': 0})['num_fn'] for cls in class_names]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(class_names))
    width = 0.25

    bars1 = ax.bar(x - width, tp_counts, width, label='True Positive', color='#00FF00', alpha=0.8)
    bars2 = ax.bar(x, fp_counts, width, label='False Positive', color='#FF0000', alpha=0.8)
    bars3 = ax.bar(x + width, fn_counts, width, label='False Negative', color='#0000FF', alpha=0.8)

    # Customize plot
    ax.set_xlabel('Classes', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Detection Results Summary by Class', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45 if len(max(class_names, key=len)) > 8 else 0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to: {save_path}")

    return fig


# drawing functions - fastai style
def bb_hw(a):
    """ convert x_min, y_min, x_max, y_max to width, height """
    # return np.array([a[1], a[0], a[3]-a[1], a[2]-a[0]])
    return np.array([a[0], a[1], a[2]-a[0], a[3]-a[1]])

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, bb, color='white'):
    patch = ax.add_patch(patches.Rectangle(bb[:2], *bb[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt,
                   verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 2)

def show_img(im, figsize=None, ax=None):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_im(im, bbs, classes, ax=None, figsize=None, title=None, color='white'):
    """ ann = ([bb], [class]) """
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax = show_img(im, figsize=figsize, ax=ax)
    # for bb, cla in ann:
    for i, bb in enumerate(bbs):
        bb = bb_hw(bb)
        draw_rect(ax, bb, color=color)
        draw_text(ax, bb[:2], classes[i], color=color)
    if title:
        ax.set_title(title)
    return ax


def plot_yolo_results(results, 
                      image: Union[str, np.ndarray, Image.Image],
                      class_name: str,
                      gt: List[Dict]):

    # Extract bounding boxes, confidence scores, and class names from the YOLOv5 results
    pred = results.xyxy[0] # results.xyxy[0] contains the detections in the format [x1, y1, x2, y2, confidence, class]

    # Extract bounding boxes (x1, y1, x2, y2), confidence scores, and class names for ALL detections
    detected_boxes_raw = pred[:, :4].cpu().numpy()
    confidence_scores_raw = pred[:, 4].cpu().numpy()
    detected_classes_indices_raw = pred[:, 5].cpu().numpy().astype(int)
    detected_classes_names_raw = [class_name[i] for i in detected_classes_indices_raw]

    ## debug
    _, ax = plt.subplots(figsize=(8, 8))

    bbs = [g['bbox'] for g in gt]
    clss = [g['class_name'] for g in gt]

    draw_im(image, bbs=bbs, classes=clss, ax=ax)
    draw_im(image, bbs=detected_boxes_raw, classes=detected_classes_names_raw, color='red', ax=ax)


