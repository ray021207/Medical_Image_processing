# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1va_b4R7FVgHL7-yGxGVAyC13YssEcKNE

Exercise 1
"""

import zipfile
import os

zip_file_path = '/path_to/Practice_PNGandJPG.zip'
extract_dir = '/content/extracted_images'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Zip file extracted to: {extract_dir}")

# List the extracted files
extracted_files = os.listdir(extract_dir)
print("\nFiles in the extracted directory:")
for file in extracted_files:
    print(file)

from PIL import Image
from IPython.display import display
import os

new_size = (128, 128)
resized_images = [] # Initialize the list
print(f"Desired new image size: {new_size}")

# Define the directory where images were extracted
# This variable should be available from a previous cell's execution
extract_dir = '/content/extracted_images/Practice_PNGandJPG' # Assuming this is the subdirectory containing images

# Get a list of all image files in the directory
image_files = []
if os.path.exists(extract_dir):
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
else:
    print(f"Error: Directory {extract_dir} not found.")


if image_files:
    print("\nDisplaying the first few images:")
    # Display only the first few images to avoid clogging the output
    for img_path in image_files[:5]:
        try:
            img = Image.open(img_path)
            resized_img = img.resize(new_size)
            resized_images.append(resized_img)
            print(f"Displaying: {img_path}")
            display(resized_img) # Display each image individually
            img.close() # Close the image file after displaying
        except Exception as e:
            print(f"Could not display {img_path}: {e}")
else:
    print("No image files found to display.")

# Commented out IPython magic to ensure Python compatibility.
# %pip install pydicom

zip_file_path = '/path_to/Practice_DICOM.zip'
extract_dicom = '/content/extracted_dicom_images'


os.makedirs(extract_dicom, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dicom)

print(f"Zip file extracted to: {extract_dicom}")

extracted_files = os.listdir(extract_dicom)
print("\nFiles in the extracted directory:")
for file in extracted_files:
    print(file)

"""Exercise 2"""

import os
import pydicom
from PIL import Image
from IPython.display import display
import numpy as np

extracted_dicom_dir = '/content/extracted_dicom_images/Practice_DICOM/chestimages_DICOM/' # Update this path if necessary

# Get a list of all DICOM files in the extracted directory
dicom_files = [os.path.join(extracted_dicom_dir, f) for f in os.listdir(extracted_dicom_dir) if f.endswith('.dcm')]

if dicom_files:
    print(f"Found {len(dicom_files)} DICOM files.")
    print("\nDisplaying the first few DICOM images:")

    # Display only the first few images to avoid clogging the output
    for dicom_path in dicom_files[:5]:
        try:
            ds = pydicom.dcmread(dicom_path)

            # Convert the DICOM pixel data to a displayable format
            # This might need adjustments based on the specific DICOM file characteristics
            if 'PixelData' in ds:
                # Convert to numpy array
                pixel_array = ds.pixel_array

                # Normalize or scale the pixel data if necessary for display
                # For simplicity, we'll just scale to 8-bit if it's not already
                if pixel_array.dtype != np.uint8:
                     # Simple scaling to 8-bit for visualization
                     pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                     pixel_array = pixel_array.astype(np.uint8)


                # Create a PIL Image from the numpy array
                # Assuming grayscale for simplicity, mode='L'
                img = Image.fromarray(pixel_array, mode='L')

                print(f"Displaying: {dicom_path}")
                display(img)
            else:
                print(f"No PixelData found in {dicom_path}")

        except Exception as e:
            print(f"Could not display {dicom_path}: {e}")
else:
    print("No DICOM files found in the specified directory.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# Set the name of your dataset zip file.
# IMPORTANT: You must first upload this file to your Google Colab session.
ZIP_FILE_NAME = "/path_to/Directions01.zip"
DATA_DIR = "Directions"
OUTPUT_DIR = "processed_images"

# Image and model parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
CLASS_NAMES = ['up', 'down', 'left', 'right'] # Ensure this matches your folder names

# --- 2. DATA PREPARATION ---
def setup_dataset():
    """
    Unzips the dataset file and sets up the directory structure.
    This function is crucial for running the code in Google Colab.
    """
    if not os.path.exists(DATA_DIR):
        print(f"Unzipping '{ZIP_FILE_NAME}'...")
        try:
            with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print("Unzipping complete. Dataset directory created.")
        except FileNotFoundError:
            print(f"Error: The file '{ZIP_FILE_NAME}' was not found.")
            print("Please upload the zip file to your Colab session and try again.")
            return False
    else:
        print("Dataset directory already exists. Skipping unzipping.")
    return True

def create_datasets():
    """
    Loads the training and validation datasets directly from the directory structure.
    The images are rescaled and categorized based on their parent folder names.
    ResNet50 requires 3 color channels, so we use 'rgb'.
    """
    if not setup_dataset():
        return None, None

    # Define paths to training and testing data
    # Corrected paths to point inside the 'Directions01' subfolder
    base_data_dir = os.path.join(DATA_DIR, 'Directions01') # Updated base directory
    train_data_dir = os.path.join(base_data_dir, 'train')
    test_data_dir = os.path.join(base_data_dir, 'test')


    # Data generators for training and validation.
    # The rescale factor divides the pixel values (0-255) by 255 to normalize them (0-1).
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Use flow_from_directory to automatically infer labels from folder names.
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        class_mode='categorical'
    )

    return train_generator, test_generator

# --- 3. MODEL ARCHITECTURE ---
def build_model(input_shape):
    """
    Constructs a model using the ResNet50 architecture for transfer learning.
    The pre-trained ResNet50 model acts as a feature extractor.
    """
    # Load the ResNet50 model with pre-trained weights from ImageNet
    # We do not include the top classification layer as we will add our own
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the base model to use its pre-trained features without modification
    base_model.trainable = False

    # Create the new model
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(CLASS_NAMES), activation='softmax') # Output layer for 4 classes
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# --- 4. TRAINING AND EVALUATION ---
def train_and_evaluate():
    """
    The main function to orchestrate the data loading, model building, and training process.
    """
    # Create the datasets
    train_ds, test_ds = create_datasets()

    if train_ds is None or test_ds is None:
        return

    # Build the model
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3) # ResNet50 expects 3 color channels
    model = build_model(input_shape)

    # Display the model summary to see the layers and parameters
    model.summary()

    # Train the model
    print("\n--- Starting model training ---")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS
    )

    # Evaluate the model on the test data
    print("\n--- Evaluating the model on the test dataset ---")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Plot the training history to visualize performance
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()

"""Exercise 3"""

import zipfile
import os

zip_file_path = '/path_to/Gender01.zip'
extract_dir = '/content/extracted_images'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Zip file extracted to: {extract_dir}")

from PIL import Image
from IPython.display import display

# Define the base directory for the extracted images
data_dir = "/content/extracted_images/Gender01"

# Define the number of images to display per class
num_images_to_display = 2

# Create a dictionary to keep track of displayed images per class
displayed_counts = {class_name: 0 for class_name in class_names}

print(f"Displaying up to {num_images_to_display} images from each class:")

dataset_to_sample = image_datasets["test"]

# Iterate through the dataset to find images of each class
for i in range(len(dataset_to_sample)):
    # Get the image path and label index
    img_path, label_index = dataset_to_sample.samples[i]
    class_name = class_names[label_index]

    # Check if we still need to display images for this class
    if displayed_counts[class_name] < num_images_to_display:
        try:
            # Open and display the image
            img = Image.open(img_path)
            print(f"Displaying {class_name} image: {os.path.basename(img_path)}")
            display(img)

            # Increment the displayed count for the class
            displayed_counts[class_name] += 1

            # Close the image file
            img.close()

        except Exception as e:
            print(f"Could not display image {img_path}: {e}")

    # If we have displayed enough images for all classes, break the loop
    if all(count >= num_images_to_display for count in displayed_counts.values()):
        break

if all(count == 0 for count in displayed_counts.values()):
    print("No images were found in the specified dataset directory.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data transforms
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 3. Load dataset
data_dir = "/content/extracted_images/Gender01"
image_datasets = {
    x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
    for x in ["train", "test"]
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=2)
    for x in ["train", "test"]
}

class_names = image_datasets["train"].classes
print("Classes:", class_names)

# 4. Model: Transfer Learning (ResNet18)
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False  # freeze backbone

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes

model = model.to(device)

# 5. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 6. Training Loop
def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print()

    return model

# 7. Train the model

trained_model = train_model(model, criterion, optimizer, 30)

# 8. Evaluation (Confusion Matrix & Report)
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # Precision, Recall, F1
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:\n", report)

# Run evaluation on test set
evaluate_model(trained_model, dataloaders["test"])

# 9. Save model
torch.save(trained_model.state_dict(), "gender_classification_resnet18.pth")
print("Model saved!")

"""EXERCISE 04

"""

import zipfile
import os

zip_file_path = '/path_to/XPAge01_RGB.zip'
extract_dir = '/content/age_images'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Zip file extracted to: {extract_dir}")

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Dataset
class AgeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]   # first column = filename
        age = torch.tensor(float(self.data.iloc[idx, 1]), dtype=torch.float32)  # second column = age
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, age

# Transforms
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Load Data
train_dataset = AgeDataset("/content/age_images/XPAge01_RGB/XP/trainingdata.csv", "/content/age_images/XPAge01_RGB/XP/JPGs", transform)
test_dataset  = AgeDataset("/content/age_images/XPAge01_RGB/XP/testdata.csv", "/content/age_images/XPAge01_RGB/XP/JPGs", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*64*64, 1)  # 128x128 input -> 64x64 after pool

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16*64*64)
        x = self.fc1(x)
        return x.squeeze()

model = SimpleCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
train_losses, test_losses = [], []

for epoch in range(10):
    # Train
    model.train()
    running_loss = 0
    for imgs, ages in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss/len(train_loader))

    # Test
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for imgs, ages in test_loader:
            outputs = model(imgs)
            loss = criterion(outputs, ages)
            running_loss += loss.item()
    test_losses.append(running_loss/len(test_loader))

    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.2f}, Test Loss: {test_losses[-1]:.2f}")

# Plot Loss
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

"""EXERCISE 05

"""

!pip install albumentations==1.3.0 -q

import os, zipfile, cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from albumentations import Compose, Resize, Normalize, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2

zip_path = "/path_to/Segmentation01.zip"
extract_path = "/content/Segmentation01"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted folders:", os.listdir(extract_path))

# 3. Dataset Class
class LungSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Join extract_path and the image_dir to get the full path
        full_image_dir = os.path.join(extract_path, "Segmentation01", image_dir)
        self.images = os.listdir(full_image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype("float32")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

# 4. Transforms
train_transform = Compose([
    Resize(256, 256),
    HorizontalFlip(p=0.5),
    Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2()
])

val_transform = Compose([
    Resize(256, 256),
    Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2()
])

# 5. U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = conv_block(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = conv_block(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = conv_block(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        c1 = self.conv1(x); p1 = self.pool(c1)
        c2 = self.conv2(p1); p2 = self.pool(c2)
        c3 = self.conv3(p2); p3 = self.pool(c3)
        c4 = self.conv4(p3); p4 = self.pool(c4)
        c5 = self.conv5(p4)

        u6 = self.up6(c5); u6 = torch.cat([u6, c4], dim=1); c6 = self.conv6(u6)
        u7 = self.up7(c6); u7 = torch.cat([u7, c3], dim=1); c7 = self.conv7(u7)
        u8 = self.up8(c7); u8 = torch.cat([u8, c2], dim=1); c8 = self.conv8(u8)
        u9 = self.up9(c8); u9 = torch.cat([u9, c1], dim=1); c9 = self.conv9(u9)

        return torch.sigmoid(self.final(c9))

# 6. Loss Function
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

# 7. Load Dataset
train_img_dir = "/content/Segmentation01/Segmentation01/train/org"
train_mask_dir = "/content/Segmentation01/Segmentation01/train/label"
val_img_dir   = "/content/Segmentation01/Segmentation01/test/org"
val_mask_dir  = "/content/Segmentation01/Segmentation01/test/label"

train_dataset = LungSegmentationDataset(train_img_dir, train_mask_dir, transform=train_transform)
val_dataset   = LungSegmentationDataset(val_img_dir, val_mask_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4)

# 8. Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
bce = nn.BCELoss()

def train_model(epochs=20):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = bce(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += (bce(preds, masks) + dice_loss(preds, masks)).item()

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

train_model(epochs=20)

# 9. Visualize Predictions
model.eval()
imgs, masks = next(iter(val_loader))
with torch.no_grad():
    preds = model(imgs.to(device)).cpu()

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 3, i*3+1)
    plt.imshow(imgs[i][0], cmap="gray"); plt.title("Image")
    plt.subplot(3, 3, i*3+2)
    plt.imshow(masks[i][0], cmap="gray"); plt.title("Mask")
    plt.subplot(3, 3, i*3+3)
    plt.imshow(preds[i][0] > 0.5, cmap="gray"); plt.title("Prediction")

plt.subplots_adjust(hspace=0.5) # Add vertical space between subplots
plt.show()

"""EXERCISE 06"""

import zipfile
import os

# Define the path to your zip file and the directory to extract to
zip_file_path = '/path_to/segmentation02.zip'  # Replace with the actual path to your zip file
extract_dir = '/content/seg_folder'  # Replace with your desired extraction directory

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Zip file extracted to: {extract_dir}")
except Exception as e:
    print(f"An error occurred: {e}")

# Import necessary libraries
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# --- Paths: change these if needed ---
base_path = '/content/seg_folder/segmentation02/segmentation'  # Root folder containing org_train, label_train, org_test, label_test
datasets = ['train', 'test']  # Datasets to process

# Pixel values in label images
PIXELS = {
    'lung': 255,
    'heart': 85,
    'outside_lung': 170,
    'outside_body': 0
}

# --- Process each dataset ---
for ds in datasets:
    org_path = os.path.join(base_path, f'org_{ds}')
    label_path = os.path.join(base_path, f'label_{ds}')

    # Output folders for extracted regions
    for region in PIXELS.keys():
        os.makedirs(os.path.join(base_path, f'{ds}_{region}'), exist_ok=True)

    # Get list of images
    images = sorted(os.listdir(org_path))

    for img_name in images:
        # Read original image
        org_img = cv2.imread(os.path.join(org_path, img_name), cv2.IMREAD_GRAYSCALE)

        # Read label image - Construct the correct label filename
        label_img_name = img_name.replace('.bmp', '_label.png') # Replace .bmp with _label.png
        label_img_path = os.path.join(label_path, label_img_name)
        label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)


        # Dictionary to hold extracted regions
        extracted = {}

        # Ensure original and label images were read correctly
        if org_img is None:
            print(f"Warning: Could not read original image {img_name}. Skipping.")
            continue
        if label_img is None:
            print(f"Warning: Could not read label image for {img_name}. Skipping.")
            continue

        # Resize label image to match original image dimensions if necessary
        if org_img.shape != label_img.shape:
            print(f"Warning: Original image {img_name} and label image have different sizes. Resizing label.")
            label_img = cv2.resize(label_img, (org_img.shape[1], org_img.shape[0]), interpolation=cv2.INTER_NEAREST)


        # Extract each region
        for region, pixel_val in PIXELS.items():
            # Create mask based on pixel value and ensure it's the same size and type as org_img
            mask = np.array(label_img == pixel_val).astype(np.uint8) * 255 # Create a binary mask (0 or 255)
            # Ensure mask is the same size as the original image (already done by resizing label_img)

            region_img = cv2.bitwise_and(org_img, org_img, mask=mask)
            extracted[region] = region_img

            # Save extracted region
            save_name = img_name.replace('.bmp', f'_{region}.png')
            save_path = os.path.join(base_path, f'{ds}_{region}', save_name)
            cv2.imwrite(save_path, region_img)

        # Optional: visualize original and all regions
        plt.figure(figsize=(12,4))
        plt.subplot(1,5,1)
        plt.title('Original')
        plt.imshow(org_img, cmap='gray')
        plt.axis('off')

        for i, region in enumerate(PIXELS.keys(), start=2):
            plt.subplot(1,5,i)
            plt.title(region)
            plt.imshow(extracted[region], cmap='gray')
            plt.axis('off')

        plt.show()

        # Optional: print pixel counts for each region
        pixel_counts = {region: np.sum((label_img==pixel_val)) for region, pixel_val in PIXELS.items()}
        print(f"{img_name}: {pixel_counts}")

import os

extracted_dir = '/content/seg_folder/segmentation02/segmentation' # Update this path if necessary

print(f"Contents of {extracted_dir}:")
print(os.listdir(extracted_dir))

# Check inside the train and test directories
for dataset in ['train', 'test']:
    org_dir = os.path.join(extracted_dir, f'org_{dataset}')
    label_dir = os.path.join(extracted_dir, f'label_{dataset}')
    print(f"\nContents of {org_dir}:")
    print(os.listdir(org_dir)[:10]) # Print only the first 10 files to keep output clean
    print(f"\nContents of {label_dir}:")
    print(os.listdir(label_dir)[:10]) # Print only the first 10 files

"""EXERCISE 07"""

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Paths ---
base_path = '/content/seg_folder/segmentation02/segmentation'
datasets = ['train', 'test']
regions_to_localize = ['lung', 'heart'] # Specify which segmented regions to localize

# Define colors for bounding boxes (BGR format)
REGION_COLORS = {
    'left_lung': (0, 255, 0),  # Green for left lung
    'right_lung': (255, 165, 0), # Orange for right lung
    'heart': (255, 0, 0)  # Blue for heart
}

# --- Localization Function ---
def localize_regions_on_original(original_image_path, segmented_mask_paths, region_colors):
    """Reads original image and segmented masks, and returns original image with bounding boxes for multiple regions."""
    org_img = cv2.imread(original_image_path, cv2.IMREAD_COLOR) # Read original in color

    if org_img is None:
        print(f"Warning: Could not read original image {original_image_path}. Skipping localization.")
        return None

    localized_img = org_img.copy() # Draw on a copy of the original image
    img_center_x = org_img.shape[1] // 2 # Get the x-coordinate of the image center

    for mask_path, region in zip(segmented_mask_paths, regions_to_localize): # Iterate through regions to get correct color
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read mask image {mask_path}. Skipping this region.")
            continue

        # Ensure mask is the same size as the original image
        if org_img.shape[:2] != mask.shape[:2]:
             mask = cv2.resize(mask, (org_img.shape[1], org_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the contours on the original image
        for cnt in contours:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(cnt)

            # Determine color based on region and position for lungs
            color = REGION_COLORS.get(region, (255, 255, 255)) # Default color

            if region == 'lung' and len(contours) > 1: # Attempt to differentiate left/right lung if multiple contours found for lung
                # Calculate the centroid of the contour
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    # Assign color based on horizontal position relative to image center
                    if cx < img_center_x:
                        color = REGION_COLORS.get('left_lung', (0, 255, 0))
                    else:
                        color = REGION_COLORS.get('right_lung', (255, 165, 0))
            elif region == 'lung' and len(contours) == 1: # If only one lung contour, use a single lung color
                 color = REGION_COLORS.get('lung', (0, 255, 0)) # Use the general lung color
            elif region in REGION_COLORS: # For other regions like heart
                 color = REGION_COLORS[region]


            # Draw the rectangle (image, start_point, end_point, color, thickness)
            cv2.rectangle(localized_img, (x, y), (x+w, y+h), color, 2) # Use determined color

    return localized_img

# --- Process and Localize ---
print("Starting combined organ localization on original images...")

for ds in datasets:
    org_dir = os.path.join(base_path, f'org_{ds}') # Path to original images
    combined_localized_output_dir = os.path.join(base_path, f'{ds}_combined_localized_on_original') # New output dir for combined images
    os.makedirs(combined_localized_output_dir, exist_ok=True)

    print(f"\nProcessing {ds} dataset for combined localization")

    # Get list of original images
    original_images = sorted(os.listdir(org_dir))

    for img_name in original_images:
        original_image_path = os.path.join(org_dir, img_name)

        # Collect paths to segmented masks for this original image
        segmented_mask_paths = []
        regions_for_image = [] # Collect region names for this image
        for region in regions_to_localize:
            segmented_mask_name = img_name.replace('.bmp', f'_{region}.png') # Construct segmented mask filename
            segmented_mask_path = os.path.join(base_path, f'{ds}_{region}', segmented_mask_name)
            if os.path.exists(segmented_mask_path): # Check if the segmented mask file exists
                 segmented_mask_paths.append(segmented_mask_path)
                 regions_for_image.append(region) # Add region name


        # Only proceed if at least one segmented mask was found for this image
        if segmented_mask_paths:
            # Pass segmented_mask_paths and regions_for_image to the localization function
            localized_img = localize_regions_on_original(original_image_path, segmented_mask_paths, regions_for_image) # Pass region names instead of colors

            if localized_img is not None:
                # Save the combined localized image
                save_name = img_name.replace('.bmp', '_combined_localized_on_original.png')
                save_path = os.path.join(combined_localized_output_dir, save_name)
                cv2.imwrite(save_path, localized_img)
                # print(f"Saved combined localized image: {save_path}")
        else:
            print(f"No segmented masks found for image: {img_name}. Skipping combined localization.")


print("\nCombined organ localization on original images complete.")

# --- Visualize some combined localized images ---
print("\nDisplaying a few combined localized images on original images:")

for ds in datasets:
    combined_localized_dir = os.path.join(base_path, f'{ds}_combined_localized_on_original')
    combined_localized_images = sorted(os.listdir(combined_localized_dir))

    # Display only the first few combined localized images
    print(f"\nCombined localized images for {ds} dataset:")
    for i, img_name in enumerate(combined_localized_images[:3]): # Display first 3
        img_path = os.path.join(combined_localized_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"{ds}: {img_name}")
            plt.axis('off')
            plt.show()
        if i == 2: # Stop after displaying 3 for this dataset
            break

"""EXERCISE 08"""

# !pip install -q timm torchvision==0.15.2

import os, zipfile, random, glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# 1) Unzip / upload dataset
from google.colab import files

print("Upload your autoencoder_img.zip (or press Cancel and mount Drive)")
zip_name = '/path_to/autoencoder_img.zip'

# if len(uploaded) == 0:
#     raise SystemExit("No file uploaded. Upload autoencoder_img.zip or mount Google Drive and set path manually.")

# zip_name = list(uploaded.keys())[0]
print("Uploaded:", zip_name)

extract_dir = "/content/autoencoder_dataset"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall(extract_dir)

print("Extracted to:", extract_dir)
# Inspect structure
for root, dirs, files in os.walk(extract_dir):
    print(root, "->", len(files), "files")

# 2) Flexible image dataset
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

all_image_paths = []
for ext in image_extensions:
    all_image_paths += glob.glob(os.path.join(extract_dir, "**", f"*{ext}"), recursive=True)

all_image_paths = sorted(list(set(all_image_paths)))
print(f"Found {len(all_image_paths)} images in archive.")

random.seed(42)
random.shuffle(all_image_paths)

# Use 70% train / 15% val / 15% test by default
n = len(all_image_paths)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)
train_paths = all_image_paths[:n_train]
val_paths   = all_image_paths[n_train:n_train+n_val]
test_paths  = all_image_paths[n_train+n_val:]

print("Train:", len(train_paths), "Val:", len(val_paths), "Test:", len(test_paths))

# Transform: grayscale, resize, normalize to [0,1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # [0,1]
])

class ImgDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("L")
        if self.transform:
            img = self.transform(img)
        # optionally return filename
        return img, os.path.basename(p)

train_ds = ImgDataset(train_paths, transform=transform)
val_ds   = ImgDataset(val_paths, transform=transform)
test_ds  = ImgDataset(test_paths, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 3) Convolutional Autoencoder (simple, effective)
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 128
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 64
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), #32
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), #16
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), #8
            nn.BatchNorm2d(512), nn.ReLU(True),
        )
        # bottleneck conv -> latent
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        # decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, stride=2, padding=1), #16
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), #32
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), #64
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), #128
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), #256
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        z = self.enc(x)
        z = self.bottleneck(z)
        out = self.dec(z)
        return out

model = ConvAutoencoder(latent_dim=256).to(DEVICE)
print(model)

# 4) Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
criterion = nn.MSELoss()

EPOCHS = 30
best_val = 1e9

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # validation reconstruction loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs)
            val_loss += criterion(preds, imgs).item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_autoencoder.pth")

    print(f"Epoch {epoch+1}/{EPOCHS}  TrainLoss: {train_loss:.6f}  ValLoss: {val_loss:.6f}")

# load best
model.load_state_dict(torch.load("best_autoencoder.pth"))
model.eval()

# 5) Compute reconstruction errors on test set
import math, statistics
recon_errors = []
filenames = []
recon_maps = {}

with torch.no_grad():
    for imgs, names in tqdm(test_loader):
        imgs = imgs.to(DEVICE)
        preds = model(imgs)
        # per-image MSE
        mse_map = ((preds - imgs)**2).mean(dim=1, keepdim=False).squeeze().cpu().numpy()  # H x W
        score = float(mse_map.mean())
        recon_errors.append(score)
        filenames.append(names[0])
        recon_maps[names[0]] = (imgs.cpu().squeeze().numpy(), preds.cpu().squeeze().numpy(), mse_map)

# Summary statistics
print("Reconstruction error stats: min {:.6f}, max {:.6f}, mean {:.6f}, median {:.6f}".format(
    np.min(recon_errors), np.max(recon_errors), np.mean(recon_errors), np.median(recon_errors)))

# 6) Thresholding and visualization
# Example threshold: percentile of validation reconstruction errors (we can compute val errors similarly)
val_errors = []
with torch.no_grad():
    for imgs, names in val_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs)
        score = float(((preds - imgs)**2).mean().item())
        val_errors.append(score)
thresh = np.percentile(val_errors, 95)  # images with error > thresh considered anomalies
print("Threshold (95th percentile on val):", thresh)

# Sort test images by reconstruction error descending (most anomalous first)
order = np.argsort(recon_errors)[::-1]

# show top-K anomalies and some normal images
K = 6
to_show = list(order[:K]) + list(order[-K:])

plt.figure(figsize=(12, 4*len(to_show)//3))
for i, idx in enumerate(to_show):
    fname = filenames[idx]
    orig, pred, mse_map = recon_maps[fname]
    orig = np.clip(orig, 0, 1)
    pred = np.clip(pred, 0, 1)
    mse_vis = (mse_map - mse_map.min()) / (mse_map.max() - mse_map.min() + 1e-8)

    ax = plt.subplot(len(to_show), 3, 3*i+1)
    ax.imshow(orig, cmap='gray'); ax.set_title(f"Orig: {fname}\nErr={recon_errors[idx]:.6f}")
    ax.axis('off')
    ax = plt.subplot(len(to_show), 3, 3*i+2)
    ax.imshow(pred, cmap='gray'); ax.set_title("Recon")
    ax.axis('off')
    ax = plt.subplot(len(to_show), 3, 3*i+3)
    ax.imshow(mse_vis, cmap='inferno'); ax.set_title("MSE map")
    ax.axis('off')

plt.tight_layout()
plt.show()

import random
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sure model is loaded and in eval mode
model.eval()

# Pick one random image from the test set
img_path = random.choice(test_paths)
print("Testing on image:", img_path)

# Load, preprocess, and move to device
from PIL import Image
img = Image.open(img_path).convert("L")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# Forward pass through autoencoder
with torch.no_grad():
    recon = model(img_tensor)

# Convert tensors to numpy
orig_np = img_tensor.cpu().squeeze().numpy()
recon_np = recon.cpu().squeeze().numpy()

# Compute per-pixel squared error
mse_map = (orig_np - recon_np) ** 2
mean_error = mse_map.mean()

print(f"Reconstruction error (mean MSE): {mean_error:.6f}")

# Normalize MSE map for visualization
mse_vis = (mse_map - mse_map.min()) / (mse_map.max() - mse_map.min() + 1e-8)

# Display all
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(orig_np, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(recon_np, cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mse_vis, cmap="inferno")
plt.title(f"Error Map\n(MSE={mean_error:.5f})")
plt.axis("off")

plt.tight_layout()
plt.show()

"""EXERCISE 09"""

import os
import zipfile
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

zip_path = "/path_to/Directions01.zip"
extract_dir = "/content/directions_data"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
print("Dataset extracted at:", extract_dir)

# --------IMAGE TRANSFORMATIONS --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=extract_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

print(f"Loaded {len(dataset)} images for clustering")

# -------- FEATURE EXTRACTION --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Identity()  # remove last classification layer
model = model.to(device)
model.eval()

features_list = []

with torch.no_grad():
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        feats = model(imgs)
        features_list.append(feats.cpu().numpy())

features = np.concatenate(features_list, axis=0)
print("Extracted features shape:", features.shape)

# -------- PCA DIMENSION REDUCTION --------
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)
print("PCA completed. Shape:", features_2d.shape)

# --------  K-MEANS CLUSTERING --------
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(features_2d)
sil_score = silhouette_score(features_2d, labels)
print(f"Clustering done. Silhouette Score: {sil_score:.3f}")

# --------  VISUALIZATION --------
plt.figure(figsize=(7, 6))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", s=25)
plt.title(f"Unsupervised Clustering of X-ray Images\n(Silhouette Score={sil_score:.3f})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, alpha=0.3)
plt.savefig("/content/task9_clustering_plot.png")
plt.show()

# -------- CLUSTER DISTRIBUTION --------
unique, counts = np.unique(labels, return_counts=True)
print("Cluster distribution:")
for u, c in zip(unique, counts):
    print(f" - Cluster {u}: {c} images")
