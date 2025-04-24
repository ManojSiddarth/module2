import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Load YOLO model for tooth detection
yolo_model = YOLO('DentalModel1.pt')

# Define the SimpleToothCNN for feature extraction (same as before)
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

# Initialize the model
feature_model = SimpleToothCNN()
feature_model.eval()

# Define the transform for cropped tooth images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Function to extract features from a single image
def extract_features(image_path):
    # Run YOLO on the input image
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

    return features

# Function to compare features
def compare_features(stored_features, extracted_features):
    # Compute the Euclidean distance between stored features and extracted features
    distances = []
    for stored in stored_features:
        for extracted in extracted_features:
            distance = np.linalg.norm(stored - extracted)
            distances.append(distance)

    # Determine if the minimum distance is below a threshold
    min_distance = min(distances)
    threshold = 0.65  # You can adjust this threshold
    if min_distance < threshold:
        return True, min_distance
    else:
        return False, min_distance

# Main function to compare stored features with extracted features
def compare_biometrics(image_path, identity):
    # Load the stored features for the identity
    feature_save_path = f'./features/{identity}_features.npy'
    if not os.path.exists(feature_save_path):
        print("No stored features found for this user.")
        return "No Match Found"

    stored_features = np.load(feature_save_path)

    # Extract features from the input image
    extracted_features = extract_features(image_path)

    # Compare features and check if there is a match
    match_found, distance = compare_features(stored_features, extracted_features)
    
    if match_found:
        return f"Match Found (Distance: {distance})"
    else:
        return f"No Match Found (Distance: {distance})"

# Run the comparison (This will be triggered from the command line with image path and identity)
if __name__ == "__main__":
    image_path = sys.argv[1]  # Path to captured image
    identity = sys.argv[2]  # Identity to verify
    match_result = compare_biometrics(image_path, identity)
    print(match_result)
