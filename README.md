# Brest_cancer_detection_model
This repository features a deep learning-based Breast Cancer Detection model using CNNs to classify IDC vs. non-IDC in histopathology images. Built with TensorFlow/Keras, it processes 5000 RGB images (50x50 px) with preprocessing, training, and evaluation to assist in cancer diagnosis
Breast Cancer Detection Using Deep Learning
This repository presents a deep learning-based approach for detecting Invasive Ductal Carcinoma (IDC) in breast histopathology images. IDC is the most common type of breast cancer, and early detection plays a crucial role in improving patient outcomes. The project leverages Convolutional Neural Networks (CNNs) to classify histopathology images as IDC-positive or IDC-negative, aiming to assist medical professionals in cancer diagnosis.

🔹 Project Overview
Dataset: The model is trained on a dataset containing 5000 RGB images of breast tissue samples, each measuring 50x50 pixels. The dataset is labeled to distinguish between IDC and non-IDC cases.

Deep Learning Model: A CNN-based classifier is implemented using TensorFlow/Keras, trained to extract features and classify the images accurately.

Preprocessing Steps: Images undergo normalization, resizing, and augmentation to enhance model performance.

📂 Repository Structure
datasets/ → Contains train, validation, and test sets.

notebooks/ → Jupyter notebooks for data preprocessing, model training, and evaluation.

models/ → Saved trained models for inference.

results/ → Performance metrics, accuracy plots, and confusion matrices.

train.py → Script for training the CNN model.
