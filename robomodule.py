import cv2
from roboflow import Roboflow

# Initialize Roboflow with API Key
rf = Roboflow(api_key="1ejKgdviav1ziLvNEqfo")  # Replace with your API key
project = rf.workspace().project("dentalai-4oiyc")  # Replace with the correct project name
model = project.version(1).model  # Version 1 of the model

# Load an image
image_path = ".\capturetest.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Run inference
response = model.predict(image_path, confidence=40, overlap=30).json()

# Draw detections
for detection in response["predictions"]:
    x, y, w, h = int(detection["x"]), int(detection["y"]), int(detection["width"]), int(detection["height"])
    label = detection["class"]
    
    # Draw bounding box
    cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
    cv2.putText(image, label, (x - w//2, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show image
cv2.imshow("Teeth Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
