MNIST Digit Classification using PyTorch
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The notebook covers dataset loading, image visualization, CNN architecture design, and model preparation for training and evaluation.

Project Overview
The goal of this project is to build a deep learning model capable of recognizing handwritten digits (0–9) from grayscale images in the MNIST dataset.

The project demonstrates:

Loading and preprocessing image datasets
Creating data pipelines with PyTorch DataLoaders
Visualizing training samples
Building a custom CNN from scratch
Preparing a model for image classification tasks
Technologies Used
Python
PyTorch
TorchVision
NumPy
Pandas
Matplotlib
TorchMetrics
tqdm
Jupyter Notebook
Dataset
The project uses the MNIST dataset, which contains:

60,000 training images
10,000 testing images
10 classes representing digits 0–9
Grayscale images of size 28 × 28 pixels
The dataset is automatically downloaded using TorchVision.

train_dataset = datasets.MNIST(
    root="dataset/",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
Project Structure
mnist-cnn-classifier/
│
├── MNIST_CNN.ipynb
├── dataset/
└── README.md
Data Loading
The dataset is loaded using PyTorch's DataLoader.

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
This allows efficient batching and shuffling during training.

Data Visualization
The notebook includes functionality to visualize handwritten digit samples from the dataset.

imshow(torchvision.utils.make_grid(images))
This helps verify that the dataset has been loaded correctly before training.

CNN Architecture
The model consists of two convolutional layers followed by a fully connected classification layer.

Layer Structure
Input (1 × 28 × 28)

Conv2D (1 → 8)
ReLU
MaxPool

Conv2D (8 → 16)
ReLU
MaxPool

Flatten

Fully Connected Layer

Output (10 Classes)
Model Definition
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            num_classes
        )
Training Configuration
Batch Size = 60
Input Channels = 1
Output Classes = 10
Image Size = 28 × 28
Learning Outcomes
This project helped me understand:

Deep Learning Fundamentals
Convolutional Neural Networks (CNNs)
Image Classification
PyTorch Dataset and DataLoader APIs
Feature Extraction using Convolution Layers
Pooling Operations
Forward Propagation in Neural Networks
Computer Vision Workflows
Future Improvements
Complete model training pipeline
Add validation dataset support
Track training and validation accuracy
Implement confusion matrix visualization
Add precision, recall, and F1-score metrics
Save and load trained models
Compare CNN performance with fully connected networks
Experiment with deeper architectures
Installation
Clone the repository:

git clone https://github.com/Anand-N-027/<repository-name>.git
cd <repository-name>
Install required packages:

pip install pandas matplotlib torch torchvision torchmetrics tqdm
Running the Project
Launch Jupyter Notebook:

jupyter notebook
Open the notebook and run all cells sequentially.

Author
Anand Narendran

GitHub: https://github.com/Anand-N-027

