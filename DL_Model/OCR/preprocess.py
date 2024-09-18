import torch
from torchvision import transforms
import yaml

def get_transform(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return transforms.Compose([
        transforms.Resize((config['data']['image_height'], config['data']['image_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)  # Add batch dimension
