import cv2
import os

def capture_teeth_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Align your teeth within the box and press the spacebar to capture.")
    print("Press 'q' to quit without capturing.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Define center ROI box where mouth should be
        box_w, box_h = w // 3, h // 4
        start_x, start_y = (w - box_w) // 2, (h - box_h) // 2
        end_x, end_y = start_x + box_w, start_y + box_h

        # Draw guide box
        frame_with_box = frame.copy()
        cv2.rectangle(frame_with_box, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame_with_box, 'Place teeth here', (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Align Teeth and Press Spacebar', frame_with_box)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Spacebar
            # Capture the entire frame
            output_dir = './pictures'
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, 'captured_teeth_image.jpg')
            cv2.imwrite(image_path, frame)  # Save the entire frame
            print(f"Image captured and saved at {image_path}")
            break
        elif key == ord('q'):  # Quit
            print("Exiting without capturing an image.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_teeth_image()
