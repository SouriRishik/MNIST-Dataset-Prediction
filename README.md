# 🧠 Digit Recognition using Feedforward Neural Networks

## 📋 Abstract

This project addresses the classic image classification challenge of handwritten digit recognition using the MNIST dataset. Leveraging a Feedforward Neural Network (FNN) built with TensorFlow and Keras, the model classifies digits from 0 to 9 with high accuracy. The data is preprocessed by normalizing pixel values and reshaping the images for compatibility. The architecture includes multiple dense layers with ReLU activation, batch normalization, dropout, and L2 regularization. The model is compiled using the Adam optimizer and categorical cross-entropy loss and trained on the MNIST dataset. It achieved an accuracy of **98.13%**, confirming the effectiveness of FNNs when properly regularized. Future work may involve hyperparameter tuning or testing on custom digit datasets.

---

## 1. 📌 Introduction

### a. Problem Statement  
Accurate digit recognition is essential for OCR systems, banking applications, and postal automation. The goal is to develop a robust classifier for handwritten digits using deep learning techniques.

### b. Objectives
- Develop a Feedforward Neural Network for digit classification.
- Evaluate model performance using standard metrics.
- Visualize predictions and analyze misclassifications.

### c. Scope
This project is limited to the MNIST dataset and focuses on FNN-based training, evaluation, and performance visualization.

---

## 2. 📊 Dataset and Preprocessing

### a. Dataset Description
- **Source**: MNIST (via `keras.datasets`)
- **Training Set**: 60,000 grayscale images (28x28 pixels)
- **Test Set**: 10,000 images
- **Labels**: Digits 0–9

### b. Preprocessing Steps
- **Normalization**: Pixel values scaled between 0 and 1.
- **Reshaping**: Each image reshaped to a 784-dimensional vector.
- **One-Hot Encoding**: Labels converted for multiclass classification.

---

## 3. 🧠 Methodology

### a. Model Architecture (Feedforward Neural Network)
- **Input Layer**: 784 units (flattened 28x28 image)
- **Hidden Layer 1**: Dense(512), ReLU, BatchNorm, Dropout(0.3)
- **Hidden Layer 2**: Dense(256), ReLU, BatchNorm, Dropout(0.3)
- **Hidden Layer 3**: Dense(128), ReLU, BatchNorm, L2 Regularization
- **Output Layer**: Dense(10), Softmax

### b. Justification
While CNNs are commonly used for image tasks, this project shows that a well-regularized dense network can also yield high performance with proper tuning. Batch normalization and dropout help improve generalization.

### c. Implementation Details
- **API**: Keras Sequential API
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Training**: `model.fit()` on training data
- **Evaluation**: `model.evaluate()` on test set

---

## 4. ⚙️ Experimental Setup

### a. Environment
- **Libraries**: TensorFlow, Keras, NumPy, Matplotlib, Seaborn
- **Platform**: Jupyter Notebook (Python)

### b. Hyperparameters
| Parameter   | Value   |
|-------------|---------|
| Optimizer   | Adam    |
| Loss        | Categorical Crossentropy |
| Epochs      | 1       |
| Batch Size  | 64      |

### c. Train-Test Split
- 60,000 training images
- 10,000 test images

---

## 5. 📈 Results

### a. Performance Metrics
- **Test Accuracy**: `0.9813` (98.13%)

### b. Visualization
- Plotted sample predictions on test digits
- Confusion matrix to evaluate misclassifications

### c. Error Analysis
- Misclassifications between similar digits (e.g., 4 vs. 9)
- Accuracy may improve with deeper models or data augmentation

---

## 6. ✅ Conclusion & Future Work

### Summary
This project demonstrates that Feedforward Neural Networks, when well-regularized and tuned, can achieve high accuracy on digit classification tasks like MNIST.

### Future Improvements
- Apply **data augmentation** to improve generalization
- Experiment with **deeper or more complex architectures** (e.g., ResNet)
- Build and deploy a **simple web UI** for real-time digit input and prediction

---

## 📎 References
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
