# Example: Prepare data for ResNet34 classification of largest object label

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class ClassificationDataset(Dataset):
    def __init__(self, df, image_folder, label2idx=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform
        self.label2idx = label2idx or {l: i for i, l in enumerate(sorted(df['LabelName'].unique()))}
        self.idx2label = {i: l for l, i in self.label2idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['ImageID']
        image_path = f"{self.image_folder}/{image_id}.jpg"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label2idx[row['LabelName']]
        if self.transform:
            image = self.transform(image)
        return image, label


# create a custom dataset
class OpenImages(Dataset):
    def __init__(self, df, image_folder):
        self.df = df
        self.unique_images = df['ImageID'].unique()
        self.root = image_folder

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, index):
        image_id = self.unique_images[index]
        image_path = f'{self.root}/{image_id}.jpg'
        image = cv2.imread(image_path, 1)  # converting to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        df = self.df.copy()
        df = df[df['ImageID'] == image_id]  # getting the row based on the index
        h, w, _ = image.shape
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = (boxes*np.array([w,h,w,h])).astype(np.uint16)
        classes = df['LabelName'].values.tolist()

        return image, boxes, classes, image_path

# Usage:
# dataset = LargestObjectClassificationDataset(largest_object_df, IMAGE_ROOT, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

mean = [0.485, 0.456, 0.406]  # change if you use different stats
std = [0.229, 0.224, 0.225]


# Now you can use this dataloader in your tools/train.py script for training a ResNet34 classifier.
def denormalize(image, mean=mean, std=std):
    image = image.clone().permute(1, 2, 0)  # [C,H,W] → [H,W,C]
    image = image * torch.tensor(std) + torch.tensor(mean)
    return image.clamp(0, 1)


def split_by_image(df, test_size=0.2, random_state=42):
    """
    Split dataframe into train/val ensuring all rows of an ImageID 
    stay in the same split.
    
    Args:
        df: pandas.DataFrame with an 'ImageID' column
        test_size: float or int, passed to train_test_split
        random_state: for reproducibility
        
    Returns:
        df_train, df_val
    """
    # get unique images
    unique_images = df['ImageID'].unique()

    # split on image IDs
    train_ids, val_ids = train_test_split(
        unique_images,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # filter original dataframe
    df_train = df[df['ImageID'].isin(train_ids)].reset_index(drop=True)
    df_val   = df[df['ImageID'].isin(val_ids)].reset_index(drop=True)
    return df_train, df_val



IM_SIZE = 224

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},  # shift_limit=0.05
            scale=(0.9, 1.1),  # scale_limit=0.1 means ±10%, so 0.9 to 1.1
            rotate=(-15, 15),  # rotate_limit=15
            p=0.5
        ),
        A.Resize(IM_SIZE, IM_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            # --- ADD THIS ---
            # This ensures boxes are not removed unless less than 10% is visible
            min_visibility=0.1
        )
    )

def get_val_transforms():
    return A.Compose([
        A.Resize(IM_SIZE, IM_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# create a custom dataset
class ObjDetDataset(Dataset):
    def __init__(self, df, image_folder="", transforms=None):
        self.df = df
        self.unique_images = df['ImageID'].unique()
        self.root = image_folder
        self.transform = transforms
        self.label2idx = {l: i for i, l in enumerate(sorted(df['LabelName'].unique()))}

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, index):
        image_id = self.unique_images[index]
        image_path = f'{self.root}/{image_id}.jpg'
        image = cv2.imread(image_path, 1)  # converting to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # df = self.df
        df = self.df[self.df['ImageID'] == image_id]  # getting the rows based on the index
        h, w, _ = image.shape
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = (boxes*np.array([w,h,w,h])).astype(np.uint16)

        classes = df['LabelName'].values.tolist()
        labels = [self.label2idx[cls] for cls in classes]

        # Albumentations format
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
        else:
            # Ensure output is tensor even if no transform
            boxes = torch.from_numpy(boxes).to(torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        return image, boxes, labels


# Custom collate function for multi-label classification
class multilabel_collate_fn:
    """
    Collate function for multi-label classification.
    Instantiate with the number of classes.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, batch):
        # batch is a list of (image, boxes, labels_per_box) tuples

        # Separate the components
        images = [item[0] for item in batch]
        boxes = [item[1] for item in batch]
        labels_per_box = [item[2] for item in batch]

        # Use the default collate function for images.
        collated_images = default_collate(images)

        # Create multi-hot labels and stack them.
        multi_hot_labels = []
        for labels in labels_per_box:
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            if len(labels) > 0:
                multi_hot[labels.unique()] = 1.0
            multi_hot_labels.append(multi_hot)
        collated_labels = torch.stack(multi_hot_labels)

        return collated_images, boxes, collated_labels

# this is a prev version
# not in use
# keep for backup
class ObjectDetectionDataset(Dataset):
    def __init__(self, df, image_dir, classes, transforms=None):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.classes = classes
        self.image_ids = df['ImageID'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['ImageID'] == image_id]

        # Read image
        image_path = f"{self.image_dir}/{image_id}.jpg"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # Get bboxes and labels
        boxes = []
        labels = []
        for _, row in records.iterrows():
            xmin = row['XMin'] * width
            xmax = row['XMax'] * width
            ymin = row['YMin'] * height
            ymax = row['YMax'] * height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index(row['LabelName']))

        # Albumentations format
        transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
        # transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
        image = transformed['image']
        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        labels = torch.tensor(transformed['class_labels'], dtype=torch.long)

        return image, boxes, labels
