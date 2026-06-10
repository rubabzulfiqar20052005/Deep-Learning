Fashion-MNIST CNN Image Classification Project

Dataset:
Fashion-MNIST containing 60,000 training images and 10,000 testing images across 10 clothing categories.

Preprocessing:

* Images normalized to range [0,1]
* Reshaped to 28x28x1 format

Data Augmentation:

* Rotation (15 degrees)
* Width Shift (10%)
* Height Shift (10%)
* Zoom (10%)
* Horizontal Flip

CNN Architecture:

* Conv2D (32) + BatchNormalization + MaxPooling
* Conv2D (64) + BatchNormalization + MaxPooling
* Conv2D (128) + BatchNormalization + MaxPooling
* Dense (256)
* Dropout (0.5)
* Softmax Output Layer

Regularisation Strategy:

* Batch Normalization used after each convolution block
* Dropout (0.5) before output layer
* EarlyStopping callback
* ReduceLROnPlateau callback

Optimizer:
Adam

Loss Function:
Categorical Crossentropy

Evaluation:

* Test Accuracy
* Confusion Matrix
* Classification Report
* Misclassified Sample Analysis

Model Saving:
The final trained model was saved as fashion_mnist_cnn.keras and successfully reloaded for inference.

Final Result:
The CNN achieved approximately 90%–94% test accuracy depending on training conditions and random initialization.
