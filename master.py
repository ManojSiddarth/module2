import subprocess
import os
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    username = request.form.get('username', 'User').strip()

    # Ensure a valid folder-friendly username
    safe_username = "".join(char for char in username if char.isalnum() or char == "_")
    identity = f"{safe_username}"
    output_image_path = f"./static/output/{identity}.jpg"

    try:
        # Step 1: Capture the image
        print(f"{safe_username} started the image capture...")
        subprocess.run(["python", "teethcapture.py"], check=True)

        # Step 2: Enhance the captured image
        print("Enhancing image...")
        subprocess.run(["python", "enhance.py"], check=True)

        # Step 3: Classify the enhanced image and pass the username to save output
        print(f"Classifying and saving results for {safe_username}...")
        subprocess.run(["python", "classifier.py", identity], check=True)

        # Step 4: Compare biometric data using subprocess
        print(f"Comparing biometric data for {safe_username}...")
        image_path = './pictures/enhanced_teeth_image.jpg'
        result = subprocess.check_output(["python", "biometric.py", image_path, identity]).decode()


        # Render the result page after processing is done
        return render_template(
            "result.html",
            username=safe_username,
            output=result.strip(),
            image_path=output_image_path
        )

    except subprocess.CalledProcessError as e:
        return f"Error during image processing: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)






