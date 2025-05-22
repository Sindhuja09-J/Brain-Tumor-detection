# 🧠 Brain Tumor Detection 

This project presents an automated system for detecting brain tumors from MRI scan images using the **YOLOv8 object detection model**. It includes a **Flask-based web interface** that allows users to upload medical images and receive real-time tumor detection results with bounding boxes and confidence scores.

---

## 🖼️ Demo

Upload your MRI scan image through the web interface and instantly view the detection results:

- Tumors are highlighted with **bounding boxes**.
- Each detection includes **confidence scores** and **tumor size estimation**.
- Processed image is rendered directly on the result page.

---

## 📌 Features

- Real-time tumor detection using YOLOv8
- Flask-based interactive web interface
- Image upload and result visualization
- Tumor localization, classification, and size estimation
- High precision with a custom-trained YOLOv8 model

---

## 🧠 Model Training Overview

- **Model**: YOLOv8n (lightweight version)
- **Framework**: PyTorch using [Ultralytics]
- **Dataset**: Custom MRI brain tumor dataset (labeled via Roboflow)
- **Epochs**: 50
- **Image size**: 640x640
- **Training Output**: Best weights saved as `best.pt`

---

## 💻 Web Application

### Tech Stack

- **Flask**: Web framework
- **OpenCV**: Image processing
- **Pillow (PIL)**: Image handling
- **YOLOv8**: Deep learning model

### Folder Structure

