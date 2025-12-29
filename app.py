from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

MODEL_PATH = "model_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


class_names = [
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Healthy",
    "Tomato Late Blight"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]

        img = Image.open(file).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        prediction = class_names[np.argmax(pred)]
        confidence = round(float(np.max(pred)) * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

