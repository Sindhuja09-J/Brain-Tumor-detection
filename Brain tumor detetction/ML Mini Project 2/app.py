from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
import os
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Folder to store uploaded images and detected images
UPLOAD_FOLDER = './uploads'
STATIC_FOLDER = './static/detected_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

model = YOLO(r'C:\Users\ruchit kumar\Desktop\New folder (2)\ML Mini Project 2\models\best.pt')  # Load your trained YOLO model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save the uploaded file
    uploaded_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(uploaded_image_path)

    # Run prediction
    results = model.predict(source=uploaded_image_path)

    # Process YOLO predictions
    predictions = []
    detected_image_filename = 'detected_' + file.filename
    detected_image_path = os.path.join(STATIC_FOLDER, detected_image_filename)
    image = cv2.imread(uploaded_image_path)

    for result in results:  # Iterate over results
        for box in result.boxes:  # Iterate over detected boxes
            class_name = result.names[int(box.cls)]  # Get class name
            confidence = float(box.conf)  # Convert confidence to a float
            box_coordinates = [float(coord) for coord in box.xyxy[0].tolist()]  # Bounding box coordinates

            # Draw bounding box on the image
            x1, y1, x2, y2 = [int(coord) for coord in box_coordinates]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate the size of the detected tumor (area of bounding box)
            width = x2 - x1
            height = y2 - y1
            size = width * height  # Size in pixels

            predictions.append({
                'class': class_name,
                'confidence': confidence,
                'box': box_coordinates,
                'size': size  # Size of the tumor
            })

    # Save the image with bounding boxes drawn to the static folder
    cv2.imwrite(detected_image_path, image)

    return render_template('result.html', predictions=predictions, detected_image_path='/static/detected_images/' + detected_image_filename)

if __name__ == '__main__':
    app.run(debug=True)
