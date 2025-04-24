import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from model import ToothClassifier  # Your CNN model

# --- CONFIGURATION ---
YOLO_MODEL_PATH = "DentalModel.pt"
CLASSIFIER_MODEL_PATH = "teeth_classifier.pt"
CAPTURED_IMAGE_PATH = "pictures/captured_teeth_image.jpg"

# --- LOAD MODELS ---
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
classifier_model = ToothClassifier()
classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location='cpu'))
classifier_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def capture_teeth_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error.")
        return None
    print("Press SPACE to capture. Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame error.")
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.imwrite(CAPTURED_IMAGE_PATH, frame)
            print(f"Captured at {CAPTURED_IMAGE_PATH}")
            break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return CAPTURED_IMAGE_PATH

def detect_teeth(image_path):
    img = cv2.imread(image_path)
    results = yolo_model(img)
    detections = results.xywh[0].numpy()
    return img, detections

def classify_teeth(img, detections):
    features = []
    for (x, y, w, h, conf, cls_id) in detections:
        x, y, w, h = int(x), int(y), int(w), int(h)
        x1, y1 = x - w//2, y - h//2
        x2, y2 = x + w//2, y + h//2
        tooth_crop = img[y1:y2, x1:x2]
        if tooth_crop.size == 0:
            continue
        input_tensor = transform(tooth_crop).unsqueeze(0)
        with torch.no_grad():
            embedding = classifier_model(input_tensor)
            embedding = F.normalize(embedding, dim=1)
        features.append(embedding.squeeze().numpy())
    return np.array(features)

def generate_toothprint():
    image_path = capture_teeth_image()
    if not image_path:
        return
    img, detections = detect_teeth(image_path)
    toothprint = classify_teeth(img, detections)
    print("Generated toothprint with", len(toothprint), "teeth")
    return toothprint
