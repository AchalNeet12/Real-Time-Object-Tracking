# Real-Time Object Tracking

## 📋 Project Overview:
Real-Time Object Tracking is a tool that uses advanced YOLO (You Only Look Once) models to detect and track objects in real-time. It allows users to process video streams or images, identifying and tracking multiple objects simultaneously with high accuracy. The app provides an intuitive interface where users can select from different YOLO models and view detected objects in live webcam feeds or uploaded files.

## 🔍 YOLO Model
- ✅ **YOLOv8n (Nano):**
- Smallest and fastest model, ideal for devices with low computational power.
- ✅ **YOLOv8s (Small):**
- Balances speed and accuracy for moderately constrained environments.
- ✅ **YOLOv8m (Medium):**
- Higher accuracy with moderate computational demand.
- ✅ **YOLOv8l (Large):**
- Highest accuracy but requires more resources.

## 📌 Dataset Used:
The project utilizes the COCO Dataset (Common Objects in Context) for object detection. This dataset includes 80 diverse object classes like humans, animals, vehicles, and furniture. It provides labeled bounding boxes, making it suitable for training YOLO models.

## 🛠️ Project Components:
- YOLO Models and Configuration:
Pre-trained YOLO models loaded dynamically based on user selection.
Input Modes:

- Webcam: Real-time video input for dynamic detection.
File Upload: Supports both images and video files for offline analysis.

- UI and Visualization:
Interactive Streamlit interface with a responsive sidebar.
Bounding boxes and labels for clear visualization of detection results.

- Pre-Processing and Post-Processing Pipelines:
Handles frame capture, model inference, and rendering in real-time.

## 🛠️ Project Components:
- YOLO Models and Configuration:
Pre-trained YOLO models loaded dynamically based on user selection.
Input Modes:

- Webcam: Real-time video input for dynamic detection.

- File Upload: Supports both images and video files for offline analysis.
  
- UI and Visualization:
Interactive Streamlit interface with a responsive sidebar.
Bounding boxes and labels for clear visualization of detection results.

- Pre-Processing and Post-Processing Pipelines:
Handles frame capture, model inference, and rendering in real-time.

##  🤖 Technology Stack:
- ✅ Frontend:
- Streamlit: Interactive and user-friendly web interface.
  HTML/CSS Styling: For custom background and sidebar design.
  
- ✅ Backend:
- YOLOv8 Model (Ultralytics): For fast and accurate object detection.
- OpenCV: For real-time video frame processing and rendering.
  
- ✅ Data Handling:
- Pillow (PIL): For image processing and loading.
- Numpy: For array manipulation and image conversion.
  
- ✅ Deployment:
 Deployable via local systems or cloud platforms like Streamlit Cloud, AWS, or Heroku.

## 📊 Model Results:
 ✅ YOLOv8n (Nano)
- Average accuracy: ~70-75%
- Processing speed: ~45-50 FPS on modern GPUs
- Best Use Case: Real-time detection on low-end hardware and edge devices.
  
 ✅ YOLOv8s (Small)
- Average accuracy: ~80-85%
- Processing speed: ~35-40 FPS on modern GPUs
- Best Use Case: Fast object detection for applications like video surveillance and mobile devices.
  
 ✅ YOLOv8m (Medium)
- Average accuracy: ~82-88%
- Processing speed: ~30-35 FPS on modern GPUs
- Best Use Case: Industrial applications, security cameras, and applications requiring a balance of accuracy and speed.
  
 ✅ YOLOv8l (Large)
- Average accuracy: ~85-90%
- Processing speed: ~20-25 FPS on modern GPUs
- Best Use Case: High-precision tasks like autonomous vehicles and advanced surveillance systems.

## 📚 Use Cases:
- Surveillance Systems:
  Monitor live camera feeds to detect suspicious activities, vehicles, or intruders.

- Autonomous Vehicles:
  Enhance navigation and safety by detecting pedestrians, vehicles, and obstacles.

- Retail Analytics:
  Detect customer movement patterns and optimize store layouts.

- Healthcare:
  Identify and track medical instruments or analyze X-rays for anomalies.

- Sports Analysis:
  Track players and objects (e.g., ball) in real-time for tactical insights.

## 📝 Future Enhancements:
- Integration with Cloud APIs:
  Enable deployment for large-scale monitoring systems using cloud services.

- Custom Model Training:
  Allow users to upload their datasets and train YOLO models for domain-specific tasks.

- Multi-Camera Support:
  Extend capabilities to process multiple camera feeds simultaneously.

- Advanced Analytics:
  Include features like object counting, heatmaps, and activity tracking.







