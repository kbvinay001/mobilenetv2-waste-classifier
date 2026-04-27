import os
import shutil
from sklearn.model_selection import train_test_split
import random

# Paths
dataset_path = "D:/Vcodez_project/dataset"
output_path = "D:/Vcodez_project/split_dataset"

# Create train/validation/test splits
train_path = os.path.join(output_path, "train")
val_path = os.path.join(output_path, "validation")
test_path = os.path.join(output_path, "test")

# Create directories
for path in [train_path, val_path, test_path]:
    os.makedirs(path, exist_ok=True)

# Split ratio: 70% train, 15% validation, 15% test
categories = os.listdir(dataset_path)

for category in categories:
    category_path = os.path.join(dataset_path, category)
    
    if not os.path.isdir(category_path):
        continue
    
    # Create category folders
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(val_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)
    
    # Get all images
    images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    # Split data
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_path, category, img))
    
    for img in val_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(val_path, category, img))
    
    for img in test_imgs:
        shutil.copy(os.path.join(category_path, img), os.path.join(test_path, category, img))
    
    print(f"{category}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

print("\nDataset split completed!")
