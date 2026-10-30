# MNIST Digit Classification using PyTorch and TorchVision

## Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch for handwritten digit classification on the MNIST dataset. The workflow includes dataset downloading, preprocessing, visualization, model development, training, and evaluation.

The project demonstrates fundamental deep learning and computer vision concepts, including:

- Image classification using PyTorch
- Working with the MNIST dataset
- Data loading with DataLoader
- Convolutional Neural Networks (CNNs)
- Training neural networks for multi-class classification
- Visualization of image datasets
- Performance evaluation using classification metrics

---

## Directory Structure

```text
.
├── notebook.ipynb
├── README.md
└── dataset/
    └── MNIST/
        ├── raw/
        └── processed/
```

---

## Dataset Description

### MNIST Dataset

The project uses the MNIST handwritten digit dataset provided through TorchVision.

The dataset contains grayscale images of handwritten digits ranging from 0 to 9.

### Dataset Characteristics

- Image Size: 28 × 28 pixels
- Channels: 1 (Grayscale)
- Classes: 10
- Training Images: 60,000
- Test Images: 10,000

### Classes

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

---

## Requirements

The project was developed using:

- Python 3.11
- PyTorch 2.2.2
- TorchVision 0.17.2
- NumPy 1.26.4
- Pandas 2.3.3
- Matplotlib 3.10.8
- TorchMetrics 1.8.2
- tqdm 4.67.1

Install dependencies:

```bash
pip install pandas matplotlib torch torchvision torchmetrics tqdm
```

---

## Data Loading

The MNIST dataset is automatically downloaded using TorchVision.

### Training Dataset

```python
train_dataset = datasets.MNIST(
    root="dataset/",
    download=True,
    train=True,
    transform=transforms.ToTensor()
)
```

### Test Dataset

```python
test_dataset = datasets.MNIST(
    root="dataset/",
    download=True,
    train=False,
    transform=transforms.ToTensor()
)
```

---

## DataLoader Configuration

PyTorch DataLoaders are used to batch and shuffle the images.

```python
batch_size = 60

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)
```

### Configuration

- Batch Size: 60
- Shuffle: Enabled
- Dataset: MNIST

---

## Data Visualization

The notebook includes functions for displaying image batches from the dataset.

```python
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
```

Example workflow:

```python
dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
```

This allows visualization of multiple handwritten digit samples during preprocessing and debugging.

---

## CNN Architecture

The model consists of two convolutional layers followed by max-pooling operations and a fully connected classifier.

### Feature Extraction Layers

```text
Input Image (1 × 28 × 28)
        │
        ├── Conv2D (1 → 8)
        ├── ReLU
        ├── MaxPool
        │
        ├── Conv2D (8 → 16)
        ├── ReLU
        └── MaxPool
```

### Classification Layers

```text
Flatten
   ↓
Linear (16 × 7 × 7 → 10)
```

### Output Classes

```text
Digit 0
Digit 1
Digit 2
Digit 3
Digit 4
Digit 5
Digit 6
Digit 7
Digit 8
Digit 9
```

---

## Model Definition

The neural network is implemented using the PyTorch `nn.Module` class.

### Components

#### Convolution Layer 1

```python
nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1
)
```

#### Convolution Layer 2

```python
nn.Conv2d(
    in_channels=8,
    out_channels=16,
    kernel_size=3,
    padding=1
)
```

#### Pooling Layer

```python
nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
```

#### Fully Connected Layer

```python
nn.Linear(
    16 * 7 * 7,
    10
)
```

---

## Forward Pass

The forward pass follows the sequence:

```text
Input
   ↓
Conv1
   ↓
ReLU
   ↓
MaxPool
   ↓
Conv2
   ↓
ReLU
   ↓
MaxPool
   ↓
Flatten
   ↓
Linear Layer
   ↓
Output Predictions
```

Implementation:

```python
x = F.relu(self.conv1(x))
x = self.pool(x)

x = F.relu(self.conv2(x))
x = self.pool(x)

x = x.reshape(x.shape[0], -1)

x = self.fc1(x)
```

---

## Model Objective

The CNN learns to classify handwritten digits into one of ten categories:

| Class | Digit |
|---------|---------|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |

---

## Training Configuration

Typical training setup includes:

- Optimizer: Adam or SGD
- Loss Function: CrossEntropyLoss
- Batch Size: 60
- Number of Classes: 10

Example:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
```

---

## Generated Outputs

### Dataset Outputs

- Downloaded MNIST Dataset
- Processed Tensor Images
- Training DataLoader
- Testing DataLoader

### Model Outputs

- Class Probabilities
- Predicted Digits
- Training Loss
- Classification Accuracy

### Visualization Outputs

- Sample Digit Images
- Batched Dataset Previews
- Prediction Examples

---

## Future Improvements

### Model Improvements

- Add additional convolutional layers
- Introduce Dropout regularization
- Add Batch Normalization
- Experiment with deeper architectures

### Evaluation Improvements

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

### Computer Vision Improvements

- Data Augmentation
- Random Rotations
- Random Shifts
- Noise Injection

### Deep Learning Improvements

- LeNet-5
- ResNet Variants
- Vision Transformers (ViTs)
- Transfer Learning Experiments

---

## Learning Outcomes

This project provides practical experience with:

- PyTorch Fundamentals
- Convolutional Neural Networks
- Handwritten Digit Recognition
- TorchVision Datasets
- DataLoader Workflows
- Neural Network Training
- Image Classification
- Deep Learning Fundamentals
- Performance Evaluation
- Computer Vision Applications

---

## License

This project is intended for educational and learning purposes.
