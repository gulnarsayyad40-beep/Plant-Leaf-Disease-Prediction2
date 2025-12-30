import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "model_tf"  # folder jahan model save hai
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Class names
# -----------------------------
class_names = [
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Healthy",
    "Tomato Late Blight"
]

# -----------------------------
# Get image path from argument
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python predict_image.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]

if not os.path.exists(img_path):
    print(f"Error: File '{img_path}' not found!")
    sys.exit(1)

# -----------------------------
# Load and preprocess image
# -----------------------------
img = Image.open(img_path).convert("RGB")  # ensure RGB
img = img.resize((128, 128))               # match training size
img_array = np.array(img) / 255.0          # normalize
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# -----------------------------
# Predict
# -----------------------------
pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]
confidence = np.max(pred) * 100

# -----------------------------
# Print result
# -----------------------------
print("Prediction:", pred_class)
print("Confidence:", round(confidence, 2), "%")

