import os
import random
from sklearn.model_selection import train_test_split

def split_data(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    train_files, test_val_files = train_test_split(files, train_size=train_ratio)
    val_files, test_files = train_test_split(test_val_files, train_size=val_ratio/(val_ratio + test_ratio))

    for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for file in file_list:
            src = os.path.join(input_dir, file)
            dst = os.path.join(split_dir, file)
            os.rename(src, dst)

if __name__ == "__main__":
    input_dir = "path/to/input/directory"
    output_dir = "data"
    split_data(input_dir, output_dir)
