import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Step 1: Load YOLO model for tooth detection
yolo_model = YOLO('DentalModel1.pt')

# Step 2: Load or define the classifier model (CNN)
class SimpleToothCNN(nn.Module):
    def __init__(self):
        super(SimpleToothCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))  # feature vector
        return x

feature_model = SimpleToothCNN()
feature_model.eval()

# Step 3: Define transform for cropped tooth images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Step 4: Run YOLO on the input image
image_path = './pictures/enhanced_teeth_image.jpg'
results = yolo_model(image_path)
result = results[0]
boxes = result.boxes.xywh.cpu().numpy()

# Load image
image = cv2.imread(image_path)
h, w, _ = image.shape

features = []

for i, (cx, cy, bw, bh) in enumerate(boxes):
    x1 = int(max(cx - bw / 2, 0))
    y1 = int(max(cy - bh / 2, 0))
    x2 = int(min(cx + bw / 2, w))
    y2 = int(min(cy + bh / 2, h))
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    crop_tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        feature_vec = feature_model(crop_tensor).squeeze().numpy()
    features.append(feature_vec)

# Step 5: Save feature set for this identity
identity = sys.argv[1] if len(sys.argv) > 1 else 'default_user'
feature_save_path = f'./features/{identity}_features.npy'
os.makedirs('./features', exist_ok=True)
np.save(feature_save_path, features)

print(f"Feature vectors saved for identity: {identity}")
