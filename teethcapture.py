import cv2
import numpy as np

def capture_image():
    # Open a connection to the default camera (0 for built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press the spacebar to capture an image of the teeth.")
    print("Press 'q' to quit without capturing.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Mirror the frame (flip horizontally)
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        h, w, _ = frame.shape

        # Zoom: Crop center and resize (2x zoom)
        zoom_factor = 2
        new_w, new_h = w // zoom_factor, h // zoom_factor
        start_x, start_y = (w - new_w) // 2, (h - new_h) // 2
        cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
        zoomed_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Display the resulting frame
        cv2.imshow('Camera - Press Spacebar to Capture', zoomed_frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Spacebar key
            # Save the mirrored & zoomed image
            image_path = '.\pictures\captured_teeth_image.jpg'
            cv2.imwrite(image_path, zoomed_frame)
            print(f"Image captured and saved as {image_path}")
            break
        elif key == ord('q'):  # 'q' key to quit
            print("Exiting without capturing an image.")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
