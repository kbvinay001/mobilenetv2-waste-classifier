import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set your dataset path
dataset_path = "D:/Vcodez_project/dataset"

# Get all categories and count images
categories = os.listdir(dataset_path)
image_counts = {}

for category in categories:
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        count = len([f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        image_counts[category] = count

# Display results
print("Dataset Distribution:")
print("-" * 40)
for category, count in image_counts.items():
    print(f"{category}: {count} images")
print("-" * 40)
print(f"Total Images: {sum(image_counts.values())}")

# Visualize distribution
plt.figure(figsize=(10, 6))
plt.bar(image_counts.keys(), image_counts.values(), color='skyblue')
plt.xlabel('Waste Categories')
plt.ylabel('Number of Images')
plt.title('Waste Dataset Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/dataset_distribution.png')
plt.show()
