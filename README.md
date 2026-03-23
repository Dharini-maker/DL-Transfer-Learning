# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Transfer Learning is a technique where a pre-trained model (trained on a large dataset such as ImageNet) is used as a starting point for a different but related task. It leverages learned features from the original task to improve learning efficiency and performance on the new task.

VGG19 is a convolutional neural network with 19 layers. It consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. In transfer learning, we typically freeze the convolutional layers and retrain the final fully connected layers to match our dataset.

## Neural Network Model
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/06757850-b127-4ffc-899c-119e75e7d581" />

## DESIGN STEPS

### STEP 1:

Import required libraries and define image transforms.

### STEP 2:

Load training and testing datasets using ImageFolder.

### STEP 3:

Visualize sample images from the dataset.

### STEP 4:

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5:

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6:

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

## PROGRAM

### Name:Dharini.S

### Register Number: 212224040072

```python
# Load Pretrained Model and Modify for Transfer Learning

model=models.vgg19(weights=VGG19_Weights.DEFAULT)


# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)


# Include the Loss function and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Dharini.S")
    print("Register Number: 212224040072")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
# Train the model
# Write your code here
train_model(model,train_loader,test_loader)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="727" height="718" alt="image" src="https://github.com/user-attachments/assets/f8485c9f-4dc9-4946-9021-65ab883f6286" />

## Confusion Matrix

<img width="642" height="570" alt="image" src="https://github.com/user-attachments/assets/2d480d12-92af-495f-8de6-f0fdd7aca85b" />

## Classification Report

<img width="542" height="247" alt="image" src="https://github.com/user-attachments/assets/08463b71-19ee-44eb-aa98-0b192d3abc8a" />

### New Sample Data Prediction

<img width="771" height="607" alt="image" src="https://github.com/user-attachments/assets/ca33f503-380a-43d2-ad23-619882963686" />

<img width="725" height="606" alt="image" src="https://github.com/user-attachments/assets/356f7eec-14ed-4a91-96e5-449d297eb6b0" />


## RESULT

The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
