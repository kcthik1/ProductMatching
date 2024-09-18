import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import os
from dataset import NutritionDataset, get_transform
import yaml

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Create the model
    model = get_model(config['num_classes'])
    model.to(device)
    
    # Create datasets and data loaders
    train_dataset = NutritionDataset(os.path.join(config['data_dir'], 'train'), 
                                     os.path.join(config['data_dir'], 'train_annotations.json'), 
                                     get_transform(train=True))
    val_dataset = NutritionDataset(os.path.join(config['data_dir'], 'val'), 
                                   os.path.join(config['data_dir'], 'val_annotations.json'), 
                                   get_transform(train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    
    # Training loop
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save the model
    torch.save(model.state_dict(), config['saved_model_path'])
    print(f"Model saved to {config['saved_model_path']}")
    
    return model

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config)