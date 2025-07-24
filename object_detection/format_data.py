import torch
import numpy as np


def format_gt(bboxes, classes, target_class_list):
    """
    Convert bounding boxes and class names into a structured format with class indices.

    Args:
        bboxes (np.ndarray): Array of shape (N, 4) containing bounding boxes.
        classes (List[str]): List of class names corresponding to each bounding box.
        target_class_list (List[str]): Ordered list of all possible class names.

    Returns:
        List[Dict]: A list of dictionaries with keys 'bbox' and 'class'.
    """
    target_class_list = [cls.lower() for cls in target_class_list]

    result = []
    for bbox, cls in zip(bboxes, classes):
        cls_lower = cls.lower()
        if cls_lower in target_class_list:
            class_index = target_class_list.index(cls_lower)
            result.append({'bbox': list(map(int, bbox)), 'class': class_index, 'class_name': cls_lower})
        else:
            raise ValueError(f"Class '{cls}' not found in target class list.")
    return result


def format_model_predictions(model_output, class_names, confidence_threshold=0.0):
    """
    Convert model output to standardized prediction format.
    
    WHY THIS FUNCTION EXISTS:
    - Different models output predictions in different formats
    - The evaluation pipeline needs a consistent format
    - We need to filter low-confidence predictions
    - Convert raw tensors/arrays to Python data types
    
    Args:
        model_output: Raw output from your model (varies by model type)
        class_names: List of class names ['background', 'truck', 'car', 'person']
        confidence_threshold: Minimum confidence to keep predictions (0.0 = keep all)
    
    Returns:
        List of prediction dictionaries in STANDARDIZED format:
        [
            {
                'bbox': [x1, y1, x2, y2],        # Bounding box coordinates
                'score': 0.85,                   # Confidence score (0-1)
                'class': 1,                      # Class ID (integer)
                'class_name': 'truck'            # Human-readable class name
            },
            ...
        ]
    """
    predictions = {} # Use a dictionary to store predictions for each images
    
    # Create mapping from model's class names to our class_names indices
    class_mapping = {}
    for our_idx, our_name in enumerate(class_names):
        our_name_lower = our_name.lower()
        for model_id, model_name in model_output.names.items():
            if model_name.lower() == our_name_lower:
                class_mapping[model_id] = our_idx
                break

    # ==========================================
    # CASE 1: YOLO-style models (YOLOv5, YOLOv8, etc.)
    # ==========================================
    if hasattr(model_output, 'pred') and model_output.pred is not None:
        print("📦 Detected YOLO-style output format")
        
        im_files = model_output.files

        # YOLO output structure:
        # model_output.pred = [tensor_for_image1, tensor_for_image2, ...]
        # Each tensor shape: (num_detections, 6) where 6 = [x1, y1, x2, y2, conf, class_id]
        # Normalize target class names to lowercase for matching
        class_names_set = set(name.lower() for name in class_names)    

        for inx, im_detections in enumerate(model_output.pred):
            image_preds = []
            print(f"   🔍 Found {len(im_detections)} raw detections")
            
            for detection in im_detections:
                # Extract the 6 values
                x1, y1, x2, y2, conf, class_id = detection[:6]
                class_id = int(class_id)
                conf = float(conf)

                # Get predicted class name and normalize
                pred_class_name = model_output.names[class_id].lower()

                if conf >= confidence_threshold and pred_class_name in class_names_set:

                    # Get our class index from the mapping
                    our_class_id = class_mapping.get(class_id, -1)
                    if our_class_id == -1:
                        continue  # skip if mapping not found (shouldn't happen due to earlier check)

                    image_preds.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'score': conf,
                        'class': our_class_id,
                        'class_name': pred_class_name
                    })
            print(f"   ✅ Kept {len(image_preds)} predictions after confidence filtering")

            predictions[im_files[inx]] = image_preds
        
    
    # ==========================================
    # CASE 2: Detectron2-style models (Faster R-CNN, etc.)
    # ==========================================
    elif hasattr(model_output, 'instances'):
        print("📦 Detected Detectron2-style output format")
        
        # Detectron2 output structure:
        # model_output.instances has separate tensors for boxes, scores, classes
        
        instances = model_output.instances
        
        # Extract data and move to CPU/numpy
        boxes = instances.pred_boxes.tensor.cpu().numpy()     # Shape: (N, 4)
        scores = instances.scores.cpu().numpy()               # Shape: (N,)
        classes = instances.pred_classes.cpu().numpy()        # Shape: (N,)
        
        print(f"   🔍 Found {len(boxes)} raw detections")
        
        for i in range(len(boxes)):
            if scores[i] >= confidence_threshold:
                predictions.append({
                    'bbox': boxes[i].tolist(),           # Convert numpy to list
                    'score': float(scores[i]),
                    'class': int(classes[i]),
                    'class_name': class_names[int(classes[i])]
                })
        
        print(f"   ✅ Kept {len(predictions)} predictions after confidence filtering")
    
    # ==========================================
    # CASE 3: Custom dictionary format
    # ==========================================
    elif isinstance(model_output, dict):
        print("📦 Detected dictionary-style output format")
        
        # Common dictionary keys: 'boxes', 'scores', 'labels'
        # This is often used by PyTorch's torchvision models
        
        if 'boxes' in model_output and 'scores' in model_output and 'labels' in model_output:
            # Extract tensors
            boxes = model_output['boxes'].cpu().numpy()
            scores = model_output['scores'].cpu().numpy()
            labels = model_output['labels'].cpu().numpy()
            
            print(f"   🔍 Found {len(boxes)} raw detections")
            
            for i in range(len(boxes)):
                if scores[i] >= confidence_threshold:
                    predictions.append({
                        'bbox': boxes[i].tolist(),
                        'score': float(scores[i]),
                        'class': int(labels[i]),
                        'class_name': class_names[int(labels[i])]
                    })
            
            print(f"   ✅ Kept {len(predictions)} predictions after confidence filtering")
        else:
            print("   ❌ Dictionary missing required keys: 'boxes', 'scores', 'labels'")
    
    # ==========================================
    # CASE 4: Raw tensor output
    # ==========================================
    elif isinstance(model_output, torch.Tensor):
        print("📦 Detected raw tensor output format")
        
        # Common tensor formats:
        # Shape: (batch_size, max_detections, 6) where 6 = [x1, y1, x2, y2, conf, class]
        # Shape: (batch_size, max_detections, 7) where 7 = [batch_idx, x1, y1, x2, y2, conf, class]
        
        print(f"   📏 Tensor shape: {model_output.shape}")
        
        # Assume batch_size=1, take first image
        detections = model_output[0]  # Shape: (max_detections, 6 or 7)
        
        print(f"   🔍 Processing {len(detections)} detection slots")
        
        for detection in detections:
            if len(detection) >= 6:
                # Handle both 6-element and 7-element formats
                if len(detection) == 7:
                    # Format: [batch_idx, x1, y1, x2, y2, conf, class]
                    _, x1, y1, x2, y2, conf, class_id = detection
                else:
                    # Format: [x1, y1, x2, y2, conf, class]
                    x1, y1, x2, y2, conf, class_id = detection[:6]
                
                # Filter by confidence and valid class
                if conf >= confidence_threshold and class_id >= 0:
                    predictions.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'score': float(conf),
                        'class': int(class_id),
                        'class_name': class_names[int(class_id)] if int(class_id) < len(class_names) else f'class_{int(class_id)}'
                    })
        
        print(f"   ✅ Kept {len(predictions)} valid predictions")
    
    # ==========================================
    # CASE 5: Unknown format
    # ==========================================
    else:
        print("❌ Unknown model output format!")
        print(f"   Type: {type(model_output)}")
        if hasattr(model_output, '__dict__'):
            print(f"   Attributes: {list(model_output.__dict__.keys())}")
        print("   Please adapt the function for your specific model")
    
    return predictions


# ==========================================
# EXAMPLE: How to adapt for YOUR specific model
# ==========================================

def format_my_custom_model_predictions(model_output, class_names, confidence_threshold=0.0):
    """
    Template for adapting to your specific model output format.
    
    STEP 1: Print and examine your model output
    STEP 2: Identify where boxes, scores, and classes are stored
    STEP 3: Extract and convert to the standard format
    """
    predictions = []
    
    # DEBUGGING: Print model output structure
    print("🔍 Debugging model output:")
    print(f"   Type: {type(model_output)}")
    print(f"   Shape/Length: {getattr(model_output, 'shape', len(model_output) if hasattr(model_output, '__len__') else 'N/A')}")
    
    # If it's a tensor, print some values
    if isinstance(model_output, torch.Tensor):
        print(f"   Sample values: {model_output[0][:5] if len(model_output) > 0 else 'Empty'}")
    
    # TODO: Replace this section with your model's specific format
    # Example for a hypothetical custom model:
    """
    if hasattr(model_output, 'my_boxes') and hasattr(model_output, 'my_scores'):
        boxes = model_output.my_boxes.cpu().numpy()
        scores = model_output.my_scores.cpu().numpy()  
        classes = model_output.my_classes.cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] >= confidence_threshold:
                predictions.append({
                    'bbox': boxes[i].tolist(),
                    'score': float(scores[i]),
                    'class': int(classes[i]),
                    'class_name': class_names[int(classes[i])]
                })
    """
    
    return predictions


# ==========================================
# TESTING AND DEBUGGING HELPERS
# ==========================================

def debug_model_output(model, sample_input, class_names):
    """
    Helper function to understand your model's output format.
    Run this ONCE to see what your model outputs.
    """
    print("🔍 DEBUGGING MODEL OUTPUT FORMAT")
    print("=" * 50)
    
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"📦 Model output type: {type(output)}")
    
    if isinstance(output, torch.Tensor):
        print(f"📏 Tensor shape: {output.shape}")
        print(f"📊 Tensor dtype: {output.dtype}")
        print(f"🎯 Sample values: {output[0][:3] if len(output) > 0 else 'Empty'}")
    
    elif isinstance(output, dict):
        print(f"🗝️  Dictionary keys: {list(output.keys())}")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
    
    elif hasattr(output, '__dict__'):
        print(f"🏷️  Object attributes: {list(output.__dict__.keys())}")
        for attr_name in output.__dict__.keys():
            attr_value = getattr(output, attr_name)
            if isinstance(attr_value, torch.Tensor):
                print(f"   {attr_name}: shape {attr_value.shape}")
    
    # Try to format predictions
    print("\n🔄 Attempting to format predictions...")
    try:
        formatted = format_model_predictions(output, class_names)
        print(f"✅ Successfully formatted {len(formatted)} predictions")
        if len(formatted) > 0:
            print(f"📋 Sample prediction: {formatted[0]}")
    except Exception as e:
        print(f"❌ Error formatting predictions: {e}")
        print("   You need to customize the format_model_predictions function")
    
    return output


# ==========================================
# USAGE EXAMPLE
# ==========================================

# To understand your model's output format:
"""
# Run this once to debug your model output
sample_batch = next(iter(your_dataloader))
sample_input = sample_batch[0][:1]  # First image only
class_names = ['background', 'truck', 'car', 'person']

debug_model_output(your_model, sample_input, class_names)
"""

# After you understand the format, use it in evaluation:
"""
for batch_idx, (images, bbs, classes, image_paths) in enumerate(dataloader):
    with torch.no_grad():
        model_output = model(images)
        
        # This function converts the raw output to standard format
        formatted_predictions = format_model_predictions(
            model_output, 
            class_names, 
            confidence_threshold=0.1  # Keep predictions with >10% confidence
        )
        
        # Now formatted_predictions is ready for evaluation!
"""
