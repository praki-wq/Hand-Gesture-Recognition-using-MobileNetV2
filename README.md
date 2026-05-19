# Hand Gesture Recognition using MobileNetV2

## Overview

This project implements a deep learning-based hand gesture recognition system using MobileNetV2 with transfer learning. The system classifies hand gestures such as FIST, ONE, PALM, and SUPER. A Flask-based web application is developed to allow users to upload images and receive predictions in real time.

---

## Features

* User authentication system (registration and login)
* Image upload for gesture prediction
* Deep learning model using MobileNetV2
* Performance evaluation using confusion matrix and accuracy metrics
* Web-based interface using Flask

---

## Technology Stack

* Python
* TensorFlow / Keras
* OpenCV
* Flask
* SQLite

---

## Project Structure

hand-gesture-recognition/
│
├── app/            # Flask application (routes, templates, uploads)
├── models/         # Trained model file
├── results/        # Output images (accuracy, confusion matrix)
├── src/            # Model training scripts
├── README.md
├── requirements.txt
└── .gitignore

---

## Model Details

* Model: MobileNetV2 (Transfer Learning)
* Input size: 128x128 grayscale images
* Number of classes: 4 (FIST, ONE, PALM, SUPER)
* Accuracy: Approximately 88%

---

## Workflow

1. Data Collection
   Images are collected for four gesture classes and organized into separate folders.

2. Data Preprocessing

   * Convert images to grayscale
   * Resize to 128x128
   * Normalize pixel values
   * Apply histogram equalization and noise reduction

3. Model Training

   * MobileNetV2 is used as the base model
   * Transfer learning is applied
   * Custom dense layers are added for classification

4. Evaluation

   * Model performance is evaluated using accuracy, classification report, and confusion matrix

5. Deployment

   * A Flask web application is used for user interaction and prediction

---

## Results

The model achieves an accuracy of approximately 88% on the test dataset.

### Confusion Matrix
<img width="1280" height="612" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/6de2f3c6-8b85-4e29-9a4b-0c180d9be7a6" />


### Accuracy and Loss
<img width="1200" height="400" alt="Accuracy and Loss" src="https://github.com/user-attachments/assets/e35a78d8-a401-4b6e-aa9f-c8ebe4158a65" />

---

## Installation

1. Clone the repository:
   git clone https://github.com/Prajwal0504-hr/hand-gesture-recognition.git

2. Navigate to the project folder:
   cd hand-gesture-recognition

3. Install dependencies:
   pip install -r requirements.txt

---

## Running the Application

cd app
python app.py

Open a browser and go to:
http://127.0.0.1:5000

---

## Usage

1. Register or log in to the application
2. Upload a hand gesture image
3. View predicted gesture and confidence score

---

## Use Cases

* Sign language recognition
* Human-computer interaction systems
* Touchless interfaces
* Accessibility tools

---

## Future Improvements

* Real-time webcam-based gesture detection
* Increase dataset size for improved accuracy
* Deploy application on cloud platforms

---

## Author

Prajwal H R
