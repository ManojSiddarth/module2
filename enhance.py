import cv2
import numpy as np

def enhance_image(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # Convert back to BGR format
    enhanced_img = cv2.merge([enhanced_gray, enhanced_gray, enhanced_gray])

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)

    # Apply bilateral filter for noise reduction
    final = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

    # Save the processed image
    cv2.imwrite(output_path, final)
    print(f"Enhanced image saved to {output_path}")

# Example usage
input_image = ".\pictures\captured_teeth_image.jpg"  # Image captured from capture.py
output_image = ".\pictures\enhanced_teeth_image.jpg"  # Processed image to be used by classify.py
enhance_image(input_image, output_image)
