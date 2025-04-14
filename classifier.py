import sys
import os
from ultralytics import YOLO

# Load the model
model = YOLO('DentalModel1.pt')

# Run the model on the image
results = model('./pictures/enhanced_teeth_image.jpg')
result = results[0]

# Now access the boxes, labels, and confidences
boxes = result.boxes.xywh  # Get the bounding box coordinates
labels = result.names  # Get the class names

print("Bounding Box Coordinates:", boxes)
print("Labels:", labels)

# Get the output directory from the command-line arguments
identity = sys.argv[1] if len(sys.argv) > 1 else 'default_user'

# Save the results in the specified directory
output_path = f"./static/output/{identity}.jpg"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.save(output_path)

print(f"Results saved to {output_path}")









# import sys
# import os
# from ultralytics import YOLO

# # Load the model
# model = YOLO('DentalModel1.pt')

# # Run the model on the image
# results = model('./pictures/enhanced_teeth_image.jpg')
# result = results[0]

# # Now access the boxes, labels, and confidences
# boxes = result.boxes.xywh  # Get the bounding box coordinates
# labels = result.names  # Get the class names

# print("Bounding Box Coordinates:", boxes)
# print("Labels:", labels)

# # Get the output directory from the command-line arguments
# identity = sys.argv[1] if len(sys.argv) > 1 else './output/default_user'

# # Save the results in the specified directory
# result.save(f".\output\{identity}.jpg")

# print(f"Results saved to {identity}")