import yaml
from train import train
from infer import load_model, infer

def main():
    config_path = 'config.yaml'
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Starting Nutrition Label Content Prediction system...")
    
    # Training
    print("Starting training...")
    train(config_path)
    print("Training completed.")
    
    # Inference
    print("Starting inference...")
    model = load_model(config_path)
    
    # Perform inference on test images
    test_images = [f for f in os.listdir(config['data']['test_dir']) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_name in test_images:
        image_path = os.path.join(config['data']['test_dir'], image_name)
        result = infer(image_path, model, config_path)
        print(f"Predicted content for {image_name}: {result}")
    
    print("Inference completed.")

if __name__ == "__main__":
    main()
