from copy import deepcopy

import torch
from torch import nn 
from torch import optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import pandas as pd
from PIL import Image
from pathlib import Path
import mlflow
from dotenv import load_dotenv
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from argparse import ArgumentParser
import math
from random import randint, random

from configuration.config import data_path, model_dir, val_transform

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("name_tag_detection")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomFixedRatioResize:
    def __init__(self, height_range = (80, 140), p = 1.0):
        self.height_range = height_range
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            self.target_height = randint(*self.height_range)  # Random height within the specified range
            w, h = img.size
            aspect_ratio = h / w
            new_w = math.ceil(self.target_height / aspect_ratio)
            return transforms.functional.resize(img, (self.target_height, new_w))
        return img  

class NameTagDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img = Image.open(data_path / row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = row["label"]

        return img, label

argument_parser = ArgumentParser()
argument_parser.add_argument("--version", type=str, help="Specify the version of the model to save.")
args = argument_parser.parse_args()

if args.version:
    version = args.version
else:
    model_files = list(model_dir.glob("best_resnet18_*.pth"))
    if model_files:
        version_numbers = [int(f.stem.split("_")[-1].replace("v", "")) for f in model_files]
        version = "v" + str(max(version_numbers) + 1) if version_numbers else "v1"
    else:
        version = "v1"
    
train_transform = transforms.Compose([
    RandomFixedRatioResize(height_range=(80, 140), p=0.7),  # Apply RandomFixedRatioResize with 70% probability
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=(3, 7), sigma=(1.0, 3.0)),
    transforms.RandomAffine(
        degrees=(-60, 60),
        translate=(0.05,0.05),
        scale=(0.9,1.1)
    ),
    transforms.RandomPerspective(
    distortion_scale=0.2,
    p=0.5
),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

#   val_transform is imported from configuration/config.py
    
train_dataset = NameTagDataset(csv_file=data_path / "train.csv", transform=train_transform)
val_dataset = NameTagDataset(csv_file=data_path / "test.csv", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze backbone, keep model only for feature extraction
for param in model.parameters():
    param.requires_grad = False
    
# Replace classifier
model.fc = nn.Linear(model.fc.in_features, 2)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=1e-3
)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=1e-3
)


epochs = 50
best_acc = 0
best_model_state_dict = None
final_precision = 0
final_recall = 0

for epoch in range(epochs):

    model.train()

    train_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            # compute the loss for validation
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # compute the accuracy for validation
            pred = outputs.argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')

    val_loss /= len(val_loader)
    mlflow.log_metric("val_loss", val_loss, step=epoch+1)
    mlflow.log_metric("val_acc", acc, step=epoch+1)
    mlflow.log_metric("val_precision", prec, step=epoch+1)
    mlflow.log_metric("val_recall", rec, step=epoch+1)
    mlflow.log_metric("train_loss", train_loss/len(train_loader), step=epoch+1)

    if acc > best_acc:

        best_acc = acc

        best_model_state_dict = deepcopy(model.state_dict())
        
        final_precision = prec
        final_recall = rec
        
if best_model_state_dict is not None:
    model.load_state_dict(best_model_state_dict)

mlflow.log_metric("final_accuracy", best_acc)
mlflow.log_metric("final_recall", final_recall)
mlflow.log_metric("final_precision", final_precision)
#log the model to mlflow
input_example = images[0].unsqueeze(0).to("cpu")
mlflow.pytorch.log_model(
    model,
    f"best_resnet18_{version}",
    input_example=input_example,
    registered_model_name="name tag detector",
    serialization_format="pickle"
)

# log data version
mlflow.log_param("data_version", "v3")

# log dataset as artifact
mlflow.log_artifact(str(data_path / "train.csv"), artifact_path="data")
mlflow.log_artifact(str(data_path / "test.csv"), artifact_path="data")

# Save the best model to local
model_save_path = model_dir / f"best_resnet18_{version}.pth"
torch.save(best_model_state_dict, model_save_path)