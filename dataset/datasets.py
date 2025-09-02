# Example: Prepare data for ResNet34 classification of largest object label

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

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

# Now you can use this dataloader in your tools/train.py script for training a ResNet34 classifier.