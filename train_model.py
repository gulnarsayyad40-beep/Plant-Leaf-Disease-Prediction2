import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -------------------------
# PARAMETERS
# -------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_cnn.keras")

# -------------------------
# TRAIN MODEL FUNCTION
# -------------------------
def train_model():
    # Check if dataset exists
    train_dir = os.path.join(os.path.dirname(__file__), 'dataset/train')
    val_dir = os.path.join(os.path.dirname(__file__), 'dataset/val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            "Dataset not found! Make sure 'dataset/train' and 'dataset/val' exist locally."
        )

    # Image generators
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = train_data.num_classes

    # CNN Model
    model = Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),

        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),

        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    # Save model
    model.save(MODEL_PATH,
    include_optimizer=False)
    print("✅ Model trained and saved as", MODEL_PATH)

# -------------------------
# LOCAL TEST
# -------------------------
if __name__ == "__main__":
    print("Starting training...")
    train_model()
