import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Path to the source folder
source_folder = r'C:\Users\Administrator\Downloads\Final_folder_for_classification'

# List all subdirectories (each representing a class)
class_folders = [os.path.join(source_folder, folder) for folder in os.listdir(source_folder)]

# Create target folders for train and test datasets
train_folder = os.path.join(source_folder, 'train')
test_folder = os.path.join(source_folder, 'test')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Split each class folder into train and test
for class_folder in class_folders:
    class_name = os.path.basename(class_folder)
    class_files = [os.path.join(class_folder, file) for file in os.listdir(class_folder)]

    # Split files into train and test sets (80-20 split)
    train_files, test_files = train_test_split(class_files, test_size=0.2, random_state=42)

    # Copy files to respective train and test folders with tqdm progress bar
    train_class_folder = os.path.join(train_folder, class_name)
    test_class_folder = os.path.join(test_folder, class_name)

    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    print(f"Copying files for class '{class_name}'...")
    for file in tqdm(train_files, desc='Copying train files'):
        shutil.copy(file, os.path.join(train_class_folder, os.path.basename(file)))

    for file in tqdm(test_files, desc='Copying test files'):
        shutil.copy(file, os.path.join(test_class_folder, os.path.basename(file)))

print("Dataset split into train and test sets.")
