import os
import shutil
import random

train_dir = "dataset/train"
val_dir = "dataset/val"

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(0.8 * len(images))
    val_images = images[split_index:]

    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in val_images:
        shutil.move(
            os.path.join(class_path, img),
            os.path.join(val_dir, class_name, img)
        )

print("✅ Dataset split completed successfully")
