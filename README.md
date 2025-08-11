# 🍑 Peach Tree Disease Classification

## 📌 Overview
This repository contains the training code, pre-trained model weights, and usage instructions for a **Convolutional Neural Network (CNN)**–based image classifier for **Peach Tree Disease Detection**. The model classifies images into four categories: Healthy, Anarsia lineatella, Grapholita molesta, and Dead Trees. The dataset and model weights are available on Kaggle: [Peach Tree Disease Dataset & Weights](https://www.kaggle.com/datasets/marslanarshad/peach-tree-disease).

## 📂 Dataset
The dataset was created by combining and reorganizing multiple raw sources into a structured format suitable for deep learning. Train / Validation / Test splits were performed using `train_test_split` from scikit-learn. Images are organized into subfolders by class, and image augmentation is applied during training to improve generalization. **Note:** Dataset structure follows the standard Keras `ImageDataGenerator` format.

## 🧠 Model Architecture
The model is a Sequential CNN built in TensorFlow/Keras with multiple convolutional layers (ReLU activation), max pooling for spatial downsampling, dropout for regularization, a dense fully connected layer, and a softmax output layer for 4-class classification.

## 🚀 Usage
**1️⃣ Clone the Repository**
```bash
git clone https://github.com/M-ArslanArshad/peach-Tree-disease-detection.git
cd peach-Tree-disease-detection
```
**2️⃣ Download Dataset from Kaggle**
```bash
kaggle datasets download marslanarshad/peach-tree-disease
unzip peach-tree-disease.zip -d data/peach_tree_disease
```
**3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
**🔹 Option A — Train from Scratch**

-`follow the .ipynb file`

This will split the dataset, train the CNN for 10 epochs, and save the model as:
- `peach_tree_disease_model.keras`
- `peach_tree_disease_weights.weights.h5`

**🔹 Option B — Use Pre-Trained Weights**
```python
from tensorflow.keras.models import load_model
model = load_model("peach_tree_disease_model.keras")
predictions = model.predict(test_generator)
```

## 📊 Results
Training for certain epochs achieved high accuracy on both training and validation sets. The model is suitable for detecting common peach tree diseases in field imagery.

## 📜 License
This project is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**. You are free to share and adapt for non-commercial purposes with appropriate credit. Full license: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 🙌 Acknowledgments
Dataset adapted from raw sources and published on Kaggle: [marslanarshad](https://www.kaggle.com/marslanarshad). All model training, weight generation, and code authored by **Muhammad Arslan**.

## 📬 Contact
For issues, improvements, or collaborations, please open a GitHub issue or contact via Kaggle.
