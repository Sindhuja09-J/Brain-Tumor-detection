import torch
from ultralytics import YOLO

# Load the trained YOLO model
# Make sure the path points to where your trained model is saved
model = YOLO(r'C:\Users\ruchit kumar\Desktop\New folder (2)\ML Mini Project 2\models\best.pt')
  # Replace 'models/best.pt' with the correct path to your model

def predict(image):
    """
    Function to predict using the YOLO model.
    This will take an image input and return the detection results.
    """
    results = model(image)  # Run model inference on the uploaded image
    return results
