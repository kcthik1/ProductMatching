import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import yaml
import os

def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image_path, confidence_threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    pred = prediction[0]
    boxes = pred['boxes']
    scores = pred['scores']
    labels = pred['labels']
    
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask].cpu().numpy()
    filtered_scores = scores[mask].cpu().numpy()
    filtered_labels = labels[mask].cpu().numpy()
    
    detections = [
        {
            'box': box.tolist(),
            'score': score.item(),
            'label': label.item()
        }
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels)
    ]
    
    return detections, image

def save_image_with_boxes(image, detections, output_path):
    draw = ImageDraw.Draw(image)
    for det in detections:
        box = det['box']
        score = det['score']
        label = det['label']
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"Label: {label}, Score: {score:.2f}", fill="red")
    image.save(output_path)

def process_folder(model, folder_path, confidence_threshold, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            detections, original_image = predict(model, image_path, confidence_threshold)
            print(f"Detected nutritional tables for {filename}:")
            for i, det in enumerate(detections):
                print(f"  Detection {i+1}: Box: {det['box']}, Score: {det['score']:.2f}, Label: {det['label']}")
            
            output_path = os.path.join(output_dir, f"output_{filename}")
            save_image_with_boxes(original_image, detections, output_path)
            print(f"Image with bounding boxes saved to: {output_path}")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = load_model(config['saved_model_path'], config['num_classes'])
    test_folder = os.path.join(config['data_dir'], 'test')
    output_dir = config['output_dir']
    
    process_folder(model, test_folder, config['confidence_threshold'], output_dir)