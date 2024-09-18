import torch
import yaml
from PIL import Image
from model import CRNN
from preprocess import get_transform, preprocess_image

def load_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = CRNN(config['model']['cnn_output_channels'], 
                 config['model']['rnn_hidden_size'], 
                 config['model']['rnn_num_layers'], 
                 config['model']['num_classes'])
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    model.eval()
    return model

def decode_prediction(prediction, charset):
    chars = charset[1:]  
    char_list = []
    for i in range(len(prediction)):
        if prediction[i] != 0 and (i == 0 or prediction[i] != prediction[i-1]):
            char_list.append(chars[prediction[i]-1])
    return ''.join(char_list)

def infer(image_path, model, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    transform = get_transform(config_path)
    image = Image.open(image_path).convert('L')
    image_tensor = preprocess_image(image, transform).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    prediction = outputs.argmax(2).squeeze(1).cpu().numpy()
    text = decode_prediction(prediction, config['data']['charset'])
    return text

if __name__ == "__main__":
    config_path = "config.yaml"
    model = load_model(config_path)
    result = infer("data/test/image.jpg", model, config_path)
    print(f"Predicted content: {result}")
