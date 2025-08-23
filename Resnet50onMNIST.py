import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Preparation & Transformations for ResNet50 ---
# ResNet50 was trained on ImageNet, which uses 3 channels (RGB)
# and a specific image size (e.g., 224x224). We need to transform
# the MNIST data to match these requirements.
# We also include normalization with ImageNet's mean and std dev.
transform = transforms.Compose([
    # Resize the 28x28 grayscale images to 224x224 to match ResNet50's input size
    transforms.Resize(224),
    # Convert the single-channel grayscale image to a 3-channel image (RGB)
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # Normalize the images using the mean and standard deviation of ImageNet
    # This is crucial for the pre-trained weights to work correctly.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transform)

print(f"Train data size (before batching): {train_data.data.size()}")
print(f"Test data size (before batching): {test_data.data.size()}")

# --- DataLoaders ---
# DataLoaders remain the same, as they handle batching for the training loops.
loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
}

# --- ResNet50 Model ---
# This is the key change from your previous CNN class.
class ResNet50_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50_MNIST, self).__init__()
        # Load the pre-trained ResNet50 model from torchvision.
        # pretrained=True loads the weights trained on ImageNet.
        self.resnet = resnet50(weights='IMAGENET1K_V1')

        # Freeze all the layers in the ResNet model. This is the "transfer learning" part,
        # where we use the model as a fixed feature extractor.
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer (the "classifier head")
        # to fit our 10-class MNIST problem.
        # The number of input features to the new layer should match the
        # output of the last convolutional block of ResNet50.
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # The forward pass simply uses the adapted ResNet model.
        # There's no need for manual flattening or multiple conv layers.
        x = self.resnet(x)
        return x

# Instantiate the new ResNet model
model = ResNet50_MNIST().to(device)
print("\n--- New Model Architecture ---")
print(model)

# --- Loss Function & Optimizer ---
# We use the same loss function and optimizer, but apply them to the new model's parameters.
# Only the parameters of the new final layer will be trained since we froze the rest.
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Training Function ---
# The training function is very similar to your original, with minor adjustments
# for the new model's forward pass.
def train(num_epochs, model, loaders):
    model.train()
    total_step = len(loaders['train'])
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_func(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

# Train the new model
num_epochs = 10   # Using fewer epochs for a quick demonstration
train(num_epochs, model, loaders)

# --- Test Function ---
def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            images = images.to(device)
            labels = labels.to(device)
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            total += labels.size(0)
            correct += (pred_y == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'\nTest Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

# Test the new model
test()

# --- Predict from test data ---
sample = next(iter(loaders['test']))
imgs, lbls = sample
actual_number = lbls[:10].numpy()

# Move images to the correct device for inference
test_output = model(imgs[:10].to(device))
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()

print(f'\nPrediction number: {pred_y}')
print(f'Actual number: {actual_number}')
