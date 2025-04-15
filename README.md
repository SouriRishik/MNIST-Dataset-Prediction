Digit Recognition using Feed Forward Neural Networks 

Abstract 

This project tackles the classic image classification problem of handwritten digit recognition using the MNIST dataset. Leveraging a Convolutional Neural Network (CNN) built using TensorFlow and Keras, the model was trained to classify digits from 0 to 9 with high accuracy. Data was preprocessed by normalizing pixel values and reshaping the images for compatibility with the CNN. The architecture included convolutional, pooling, and dense layers, and the model was compiled using the Adam optimizer with categorical cross-entropy loss. Training and testing were performed on the standard MNIST train-test split. Performance was evaluated using accuracy and confusion matrices. The model achieved an accuracy of approximately None, confirming CNN's effectiveness for image-based tasks. Future improvements could include hyperparameter tuning or extending to custom digit datasets. 

1. Introduction 

a. Problem Statement 
Accurate digit recognition is crucial in OCR systems, banking, and postal automation. The task is to build a robust classifier for handwritten digits using machine learning. 

b. Objectives 
- Develop a CNN-based model for digit classification 
- Evaluate model performance using standard metrics 
- Visualize results and analyze misclassifications 

c. Scope 
This project uses the MNIST dataset and CNNs for digit classification. It focuses on model training, evaluation, and performance visualization. 

2. Dataset and Preprocessing 

a. Dataset Description 
- Source: MNIST dataset via keras.datasets 
- Size: 60,000 training and 10,000 testing grayscale 28x28 images 
- Structure: Each image is labeled with the correct digit (0–9) 

 

 

b. Preprocessing Steps 
- Normalization: Divided pixel values by 255 
- Reshaping: Input reshaped to (28, 28, 1) to fit CNN 
- One-hot encoding: Labels transformed for categorical classification 

 

3. Methodology 

a. Machine Learning Model(s) Used 

The project employs a Feedforward Neural Network (FNN) using Keras's Sequential API. The model comprises multiple dense (fully connected) layers, along with regularization techniques to prevent overfitting. The architecture is as follows: 

Input Layer: Accepts a flattened 784-dimensional vector (from 28x28 pixel images). 

Hidden Layer 1: Dense layer with 512 neurons and ReLU activation, followed by batch normalization and a 30% dropout layer. 

Hidden Layer 2: Dense layer with 256 neurons and ReLU activation, again followed by batch normalization and dropout. 

Hidden Layer 3: Dense layer with 128 neurons, ReLU activation, batch normalization, and L2 activity regularization. 

Output Layer: Dense layer with 10 neurons and softmax activation for multiclass classification. 

 

b. Justification 

Dense neural networks are effective for structured data such as flattened image vectors. The chosen architecture includes: 

ReLU activation for non-linearity, 

Batch normalization for stabilizing training and improving performance, 

Dropout for regularization, and 

L2 regularization to further reduce overfitting. 

Although CNNs are typically more efficient for image recognition, this model demonstrates that even a well-regularized dense network can achieve strong performance on MNIST. 

c. Implementation Details 

Model Construction: Built using Keras Sequential API with layered additions. 

Compilation: The model is compiled using the Adam optimizer and sparse categorical cross-entropy as the loss function (suitable for integer labels). 

Training: Executed using model.fit() with training data. 

Evaluation: Accuracy was measured using model.evaluate() on the test dataset. 

4. Experimental Setup 

a. Hardware/Software Used 
- Libraries: TensorFlow, Keras, NumPy, Matplotlib, Seaborn 
- Environment: Python Jupyter Notebook 

b. Hyperparameters 
- Optimizer: Adam 
- Loss: Categorical Crossentropy 
- Epochs: 1 
- Batch Size: 64 

c. Train-Test Split 
- Standard MNIST split (60K train, 10K test) 

5. Results and Screenshots of UI 

a. Performance Metrics 
- Accuracy: 0.9813 

 

b. Visualization of Results 
- Displayed predictions on sample test digits 

          

c. Error Analysis 
- Some confusion observed between visually similar digits (e.g., 4 & 9) 
- Errors may be reduced by increasing model depth or training data variety 

6. Conclusion & Future Work 

- Summary: CNN effectively classifies MNIST digits with high accuracy 

- Future Work: 
  - Use data augmentation 
  - Try deeper architectures (ResNet, etc.) 
  - Deploy model with UI for user input 
