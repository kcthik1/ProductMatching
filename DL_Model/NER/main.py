import yaml
from preprocess import preprocess_dataset
from train import train_model
from infer import process_folder
import os

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Preprocess the dataset
    preprocess_dataset(config)
    
    # Train the model
    model = train_model(config)
    
    # Perform inference on the test set
    test_folder = os.path.join(config['data_dir'], 'test')
    output_dir = config['output_dir']
    
    process_folder(model, test_folder, config['confidence_threshold'], output_dir)

if __name__ == "__main__":
    main()
