[# Image-Classification
🖼️ Image Classification on CIFAR-10

This project implements an image classification model using the CIFAR-10 dataset. CIFAR-10 is a widely used benchmark dataset consisting of 60,000 32x32 color images in 10 different classes. The goal of this project is to train a deep learning model to accurately classify images into one of the following categories:

🚗 Automobile | 🐱 Cat | 🐶 Dog | ✈️ Airplane | 🚢 Ship | 🚚 Truck | 🐸 Frog | 🐦 Bird | 🐴 Horse | 🧸 Deer

📂 Dataset

CIFAR-10 contains:

50,000 training images

10,000 testing images

Image size: 32x32x3 (RGB)

10 classes (mutually exclusive)

Dataset is available through:

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

🛠️ Tech Stack

Python

TensorFlow / Keras (for deep learning models)

NumPy, Pandas (data preprocessing & analysis)

Matplotlib, Seaborn (visualizations)

Streamlit (deployment & UI)

🚀 Model Architecture

A Convolutional Neural Network (CNN) was used for classification.

To run the application visit: https://image-classification-aopvbnrnaezmiyldjpxwuk.streamlit.app/
