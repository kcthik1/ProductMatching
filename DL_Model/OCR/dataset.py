import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import yaml

class NutritionLabelDataset(Dataset):
    def __init__(self, root_dir, config_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [img for img in os.listdir(root_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.charset = self.config['data']['charset']
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        
        # Load labels
        self.labels = []
        for img in self.images:
            label_file = os.path.splitext(img)[0] + '.txt'
            with open(os.path.join(root_dir, label_file), 'r') as f:
                self.labels.append(f.read().strip())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        label_indices = [self.char_to_idx[c] for c in label]
        
        return image, torch.LongTensor(label_indices)

    def collate_fn(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        
        # Pad sequences to max length in batch
        max_length = max(len(label) for label in labels)
        padded_labels = torch.full((len(labels), max_length), self.char_to_idx[''])
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label
        
        return images, padded_labels
