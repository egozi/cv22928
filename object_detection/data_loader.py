import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def stack_images_with_padding(images):
    """
    Stack images by padding them to the same size (preserves aspect ratio).

    Args:
        images: List of image tensors with potentially different sizes

    Returns:
        4D tensor of stacked images
    """
    # Find maximum dimensions
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded_images = []

    for img in images:
        h, w = img.shape[-2:]

        # Calculate padding
        pad_h = max_h - h
        pad_w = max_w - w

        # Pad image (left, right, top, bottom)
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded_img)

    return torch.stack(padded_images, dim=0)


def collate_fn(batch):
    images = [item[0] for item in batch]
    bbs = [item[1] for item in batch]
    classes = [item[2] for item in batch]
    image_paths = [item[3] for item in batch]

    # Depending on how the model expects input, you might need to
    # pad or resize images and bounding boxes here.
    # For now, we'll return them as lists.

    return images, bbs, classes, image_paths


# Alternative collate function with padding instead of resizing
def collate_fn_with_padding(batch):
    """
    Collate function that pads images to same size instead of resizing.
    Better preserves original image content.
    """
    images = [item[0] for item in batch]
    bbs = [item[1] for item in batch]
    classes = [item[2] for item in batch]
    image_paths = [item[3] for item in batch]

    # Convert images to tensors
    processed_images = []

    for img in images:
        # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            if hasattr(img, 'shape'):  # numpy array
                img = torch.from_numpy(img)
            else:  # PIL Image
                img = transforms.ToTensor()(img)

        # Ensure image has correct format (C, H, W)
        if img.dim() == 2:  # Grayscale
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[2] == 3:  # (H, W, C) -> (C, H, W)
            img = img.permute(2, 0, 1)

        # Ensure float type and normalize
        if img.dtype != torch.float32:
            img = img.float()
        if img.max() > 1.0:
            img = img / 255.0

        processed_images.append(img)

    # Stack with padding
    images_tensor = stack_images_with_padding(processed_images)

    return images_tensor, bbs, classes, image_paths

