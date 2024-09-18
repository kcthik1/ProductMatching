import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class NutritionDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        
        boxes = torch.as_tensor(self.annotations[img_id]['boxes'], dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # All boxes are nutritional tables
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transform:
            img, target = self.transform(img, target)
        
        return img, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

def get_transform(train):
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)