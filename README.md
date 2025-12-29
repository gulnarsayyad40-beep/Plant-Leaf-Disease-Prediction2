🌿 Plant Leaf Disease Prediction System

A Deep Learning + Flask web application that predicts plant leaf diseases from uploaded leaf images.
The project includes:

Flask backend

CNN-based deep learning model for image classification

Attractive HTML UI with templates

API support for predictions

🚀 Features

✔ Predicts leaf disease from uploaded images
✔ Shows confidence score along with predicted disease
✔ Clean and responsive UI
✔ API endpoint for Postman testing
✔ Well-structured Flask project

🧠 Tech Stack

Frontend:

HTML

CSS

Backend:

Python

Flask

Deep Learning:

TensorFlow / Keras

Convolutional Neural Network (CNN)

Other Libraries:

Pillow (PIL) for image preprocessing

NumPy

📂 Project Structure
plant-leaf-disease-prediction/
├── app.py                  # Flask application
├── predict_image.py        # Image preprocessing & prediction
├── train_model.py          # CNN model training script
├── split_dataset.py        # Dataset splitting script
├── model_cnn.keras         # Saved trained CNN model
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html          # Home page
└── static/
    └── (optional images/css)

📥 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/plant-leaf-disease-prediction.git
cd plant-leaf-disease-prediction

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Run the App
python app.py


Open the browser:

http://127.0.0.1:5000

🧪 API (Postman) Testing
POST Request

URL:

http://127.0.0.1:5000/predict


JSON Body Example:

{
  "image_path": "path_to_leaf_image.jpg"
}


Response Example:

{
  "predicted_disease": "Potato Late Blight",
  "confidence": "92%"
}

🗄 Data Flow (Step-by-Step)

Dataset Preparation:
split_dataset.py splits raw leaf images into training, validation, and test sets.

Model Training:
train_model.py trains a CNN model on the prepared dataset and saves it as model_cnn.keras.

Web Interface:
app.py + templates/ provide the user interface to upload images.

Image Prediction:
predict_image.py preprocesses images and predicts the leaf disease using the trained CNN model.

Result Display:
Prediction and confidence are displayed on a separate result page.

🧑‍💻 Author

Gulnar Sayyad