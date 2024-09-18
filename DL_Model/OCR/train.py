import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from dataset import NutritionLabelDataset
from preprocess import get_transform
from model import CRNN

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    transform = get_transform(config_path)
    train_dataset = NutritionLabelDataset(config['data']['train_dir'], config_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              shuffle=True, collate_fn=train_dataset.collate_fn)

    # Initialize model
    model = CRNN(config['model']['cnn_output_channels'], 
                 config['model']['rnn_hidden_size'], 
                 config['model']['rnn_num_layers'], 
                 config['model']['num_classes']).to(device)

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], 
                           weight_decay=config['training']['weight_decay'])

    # Training loop
    for epoch in range(config['training']['num_epochs']):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            output_lengths = torch.full(size=(outputs.size(1),), 
                                        fill_value=outputs.size(0), 
                                        dtype=torch.long)
            label_lengths = torch.sum(labels != 0, dim=1)
            
            loss = criterion(outputs, labels, output_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, "
                      f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), config['paths']['model_save_path'])
    print("Training completed. Model saved.")

if __name__ == "__main__":
    train("config.yaml")
