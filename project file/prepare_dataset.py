import os
import shutil
import random

source_root = 'data/train'  # Make sure your TRAIN images are here
target_root = 'data'
train_split = 0.8

categories = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Create test folder structure
for category in categories:
    os.makedirs(os.path.join(target_root, 'test', category), exist_ok=True)

# Move 20% of images from train to test
for category in categories:
    full_path = os.path.join(source_root, category)
    images = [img for img in os.listdir(full_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0:
        print(f"⚠️ No images found in {full_path}, skipping...")
        continue

    random.shuffle(images)
    test_count = int(len(images) * (1 - train_split))
    test_images = images[:test_count]

    for img in test_images:
        src = os.path.join(full_path, img)
        dst = os.path.join(target_root, 'test', category, img)
        
        if os.path.exists(src):
            shutil.move(src, dst)

print("\n✅ 20% of images moved to data/test/ for each category.")
