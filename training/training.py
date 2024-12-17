import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

train_losses = []
test_accuracies = []
class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)", 
    "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", 
    "No overtaking", "No overtaking vehicles greater than 3.5 tons", "Right-of-way at the next intersection", 
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 tons prohibited", 
    "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right", 
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", 
    "Road work", "Traffic signals", "Pedestrians", "Children", "Bicycles", "Beware of ice/snow", 
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", 
    "Turn left ahead", "Ahead only", "Go straight or turn right", "Go straight or turn left", 
    "Keep right", "Keep left", "Roundabout mandatory", "End of priority road", "Prior to pedestrian crossing"
]


# Define hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 20
num_classes = 43  # GTSRB has 43 traffic sign classes

# Define data transformations for training and testing
# These include resizing the images and normalizing the pixel values
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize all images to 32x32
    transforms.ToTensor(),        # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Load the GTSRB dataset
train_dataset = datasets.GTSRB(
    root="./data", split="train", transform=transform, download=True
)
test_dataset = datasets.GTSRB(
    root="./data", split="test", transform=transform, download=True
)

# Create data loaders for batching
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Second convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, num_classes)  # Fully connected layer 2
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization

    def forward(self, x):
        x = self.relu(self.conv1(x))  # Apply first conv layer and ReLU
        x = self.pool(x)             # Apply pooling
        x = self.relu(self.conv2(x)) # Apply second conv layer and ReLU
        x = self.pool(x)             # Apply pooling
        x = x.view(x.size(0), -1)    # Flatten the tensor
        x = self.relu(self.fc1(x))   # Apply first fully connected layer
        x = self.dropout(x)          # Apply dropout
        x = self.fc2(x)              # Apply final fully connected layer
        return x


# Initialize the model, loss function, and optimizer
model = CNN(num_classes=num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_accuracy():
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Plot the learning curves
def plot_learning_curves():
    plt.figure(figsize=(12, 5))

    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Training loop
def train():
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss=running_loss / len(train_loader)
        train_losses.append(avg_loss)

        test_accuracy = evaluate_accuracy()
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


# Testing loop
def test():
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

# confusion matric of the data
def plot_confusion_matrix():
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_weight_distribution():
    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.hist(param.data.cpu().numpy().flatten(), bins=50, alpha=0.6, label=name)

    plt.legend()
    plt.title("Weight Distributions")
    plt.show()




def plot_predictions(num_samples=5):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))  # Convert from tensor to image

        plt.title(f"{class_names[labels[i]]==class_names[predicted[i]]}")
        plt.axis('off')
    plt.show()




train()   # train the model
test()    # test the model on the test dataset and print the test accuracy

plot_learning_curves()  # plot the test and training acuuracy over epochs 

plot_confusion_matrix() # plot the confusion matrix

plot_weight_distribution() # plot the model weights distribution

plot_predictions(5) # plot the predictions of the first 5 images

