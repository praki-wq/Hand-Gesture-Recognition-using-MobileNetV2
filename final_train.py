import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

# =====================================
# Configuration
# =====================================
np.random.seed(42)
tf.random.set_seed(42)

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 4
CLASS_NAMES = ['FIST', 'ONE', 'PALM', 'SUPER']
DATA_PATH = "Dataset/"

# =====================================
# Data Loading
# =====================================
def load_and_preprocess_data(data_path):
    images, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found")
            continue

        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img = cv2.equalizeHist(img)
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    img = img.astype('float32') / 255.0
                    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
                    images.append(img)
                    labels.append(class_idx)
    return np.array(images), np.array(labels)



# =====================================
# MobileNetV2 Transfer Learning Model
# =====================================
def create_mobilenet_model(input_shape):
    base_model = keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),  # expects 3 channels
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze backbone

    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(3, (3, 3), padding='same'),  # convert 1-channel → 3-channel
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# =====================================
# Plots
# =====================================
def plot_training_history(history, title_prefix="Model"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title(f'{title_prefix} Accuracy'); ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'{title_prefix} Loss'); ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    return cm


# =====================================
# Main
# =====================================
def main():
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(DATA_PATH)
    if len(X) == 0:
        print("No images found. Check dataset path.")
        return

    print(f"Loaded {len(X)} images across {len(np.unique(y))} classes")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )

    
    # ---- MobileNetV2 ----
    print("\n=== Training MobileNetV2 (Transfer Learning) ===")
    mobilenet_model = create_mobilenet_model((IMG_HEIGHT, IMG_WIDTH, 1))
    mobilenet_model.compile(optimizer=keras.optimizers.Adam(0.0005),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
    history_mobilenet = mobilenet_model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                            validation_data=(X_val, y_val),
                                            epochs=EPOCHS, class_weight=class_weights,
                                            verbose=1)
    mobilenet_model.save("mobilenet.h5")
    plot_training_history(history_mobilenet, title_prefix="MobileNetV2")
    y_pred_mobilenet = np.argmax(mobilenet_model.predict(X_test), axis=1)
    print("\nMobileNetV2 Classification Report:\n",
          classification_report(y_test, y_pred_mobilenet, target_names=CLASS_NAMES))
    plot_confusion_matrix(y_test, y_pred_mobilenet, CLASS_NAMES, "MobileNetV2 Confusion Matrix")

    
    print("\n" + "="*50)
    print("ALL TRAINING COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()
