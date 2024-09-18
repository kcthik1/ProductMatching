import os
import shutil
import random
import json
import yaml

def preprocess_dataset(config):
    data_dir = config['data_dir']
    annotation_file = config['annotation_file']
    train_split = config['train_split']
    val_split = config['val_split']
    test_split = config['test_split']
    
    # Load existing annotations
    with open(os.path.join(data_dir, annotation_file), 'r') as f:
        annotations = json.load(f)
    
    # Get all image files from annotations
    image_files = list(annotations.keys())
    random.shuffle(image_files)
    
    # Calculate split indices
    total_images = len(image_files)
    train_end = int(total_images * train_split)
    val_end = train_end + int(total_images * val_split)
    
    # Create directories for train, val, and test sets
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)
    
    # Split and move files
    train_annotations = {}
    val_annotations = {}
    test_annotations = {}
    
    for i, img in enumerate(image_files):
        src = os.path.join(data_dir, img)
        if i < train_end:
            dst = os.path.join(data_dir, 'train', img)
            train_annotations[img] = annotations[img]
        elif i < val_end:
            dst = os.path.join(data_dir, 'val', img)
            val_annotations[img] = annotations[img]
        else:
            dst = os.path.join(data_dir, 'test', img)
            test_annotations[img] = annotations[img]
        
        shutil.copy(src, dst)
    
    # Save split annotations
    with open(os.path.join(data_dir, 'train_annotations.json'), 'w') as f:
        json.dump(train_annotations, f)
    with open(os.path.join(data_dir, 'val_annotations.json'), 'w') as f:
        json.dump(val_annotations, f)
    with open(os.path.join(data_dir, 'test_annotations.json'), 'w') as f:
        json.dump(test_annotations, f)

    print(f"Dataset split complete. Images and annotations divided into train, val, and test sets.")
    print(f"Train set: {len(train_annotations)} images")
    print(f"Validation set: {len(val_annotations)} images")
    print(f"Test set: {len(test_annotations)} images")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    preprocess_dataset(config)
    print("Dataset preprocessing completed.")