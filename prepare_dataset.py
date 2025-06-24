import os, shutil, random

source_root = 'data/train'  # Make sure your TRAIN images are here
target_root = 'data'
train_split = 0.8

categories = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Create empty test folders
for category in categories:
    os.makedirs(os.path.join(target_root, 'test', category), exist_ok=True)

# Move 20% from train to test
for category in categories:
    full_path = os.path.join(source_root, category)
    images = os.listdir(full_path)
    random.shuffle(images)
    test_count = int(len(images) * 0.2)
    test_images = images[:test_count]

    for img in test_images:
        src = os.path.join(full_path, img)
        dst = os.path.join(target_root, 'test', category, img)
        shutil.move(src, dst)

print("✅ Created data/test/ with 20% of images from data/train/")
