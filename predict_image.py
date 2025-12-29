import tensorflow as tf
import numpy as np
from PIL import Image
import sys

model = tf.keras.models.load_model("model_cnn.h5")

class_names = [
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Healthy",
    "Tomato Late Blight"
]

img_path = sys.argv[1]

img = Image.open(img_path).resize((224,224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]
confidence = np.max(pred) * 100

print("Prediction:", pred_class)
print("Confidence:", round(confidence, 2), "%")
