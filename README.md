# Implementation-of-Transfer-Learning

## Aim

To implement **Transfer Learning** for image classification using the **VGG-19** pre-trained model.

---

## Problem Statement and Dataset

Image classification from scratch requires a huge dataset and long training times. To overcome this, **transfer learning** can be applied using pre-trained models like **VGG-19**, which has already learned feature representations from a large dataset (**ImageNet**).

* **Problem Statement:**
  Build an image classifier using **VGG-19** pre-trained architecture, fine-tuned for a custom dataset (e.g., CIFAR-10, Flowers dataset, or any small image dataset).
* **Dataset:**
  A dataset consisting of multiple image classes (e.g., `train`, `test`, and `validation` sets). For example, CIFAR-10 (10 classes of small images) or a custom dataset with multiple classes.

---

## DESIGN STEPS

### STEP 1:

Import the required libraries (PyTorch, torchvision, matplotlib, etc.) and set up the device (CPU/GPU).

### STEP 2:

Load the dataset (train and test). Apply transformations such as resizing, normalization, and augmentation. Create `DataLoader` objects.

### STEP 3:

Load the pre-trained VGG-19 model from `torchvision.models`. Modify the **final fully connected layer** to match the number of classes in the dataset.

### STEP 4:

Define the **loss function** (CrossEntropyLoss) and the **optimizer** (Adam).

### STEP 5:

Train the model for the required number of epochs while recording **training loss** and **validation loss**.

### STEP 6:

Evaluate the model using a **confusion matrix**, **classification report**, and test it on **new samples**.

---

## PROGRAM

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset (example: CIFAR-10 or custom dataset)
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(pretrained=True)

# Freeze all layers except classifier
for param in model.features.parameters():
    param.requires_grad = False

num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader, test_loader, num_epochs=5):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss/len(test_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=10)
```

---

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot


<img width="758" height="729" alt="{9884B91C-A6A1-4785-B167-728D0C29CD8A}" src="https://github.com/user-attachments/assets/01950b2b-6e2b-4e38-bfcf-f6dd4ff42353" />


### Confusion Matrix

<img width="694" height="567" alt="{2D9BF55C-0ABB-4440-B753-28563263CE82}" src="https://github.com/user-attachments/assets/369cb12e-0b71-4c3f-a83c-e983da5e6463" />




### Classification Report

<img width="505" height="205" alt="{709BBD56-AC7F-444A-AF3B-985FD6C9BB60}" src="https://github.com/user-attachments/assets/8c043ab6-606e-4340-940a-da2158b8ae77" />



### New Sample Prediction

<img width="311" height="368" alt="{2E3808F0-9529-4259-A13B-27ACFB095AE9}" src="https://github.com/user-attachments/assets/68406d46-7fb4-49e6-8a50-2835627df405" />

<img width="322" height="375" alt="{52700A43-AEFF-498B-80F2-817C1C681D84}" src="https://github.com/user-attachments/assets/b0e5e0a1-f129-4238-b3d8-163070d494e3" />


---

## RESULT

Thus, **Transfer Learning** using **VGG-19** was successfully implemented for image classification. The model was fine-tuned on the given dataset, and evaluation using confusion matrix, classification report, and predictions on new samples confirmed its effectiveness. Transfer learning significantly reduced training time and improved accuracy compared to training from scratch.


