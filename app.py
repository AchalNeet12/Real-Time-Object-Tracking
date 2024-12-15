import random
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile
import base64

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}"); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: white;
    }}
    .sidebar .sidebar-content {{
        background-color: rgba(0, 0, 0, 0.7);
        padding: 10px;
        border-radius: 10px;
    }}
    .sidebar-box {{
        border: 2px solid white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
        background-color: rgba(255, 255, 255, 0.1);
        text-align: center;
    }}
    .sidebar-box h3 {{
        color: #FFD700;
        margin-bottom: 10px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background (provide the correct image path)
set_background("appimg.webp")

# Load YOLO Models
yolo_models = {
    "YOLOv8n (Nano)": "weights/yolov8n.pt",
    "YOLOv8s (Small)": "weights/yolov8s.pt",
    "YOLOv8m (Medium)": "weights/yolov8m.pt",
    "YOLOv8l (Large)": "weights/yolov8l.pt",
}

# Sidebar Layout
st.sidebar.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
st.sidebar.title("🔍 YOLO Object Detection")
selected_model = st.sidebar.selectbox("Select YOLO Model:", list(yolo_models.keys()))
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Add informative and styled sections to the sidebar
st.sidebar.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
st.sidebar.header("💡 Instructions")
st.sidebar.write(
    """
    - Select a YOLO model to use for detection.  
    - Choose **Webcam** or **Upload File** as input.  
    - For images or videos, view detected objects instantly!  
    """
)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Load YOLO Model
model = YOLO(yolo_models[selected_model])

# Load COCO class names
with open(r"E:\1.VS-Code Folder\YOLO\yolo\coco.txt", "r") as my_file:
    class_list = my_file.read().splitlines()

# Generate random colors for each class
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(len(class_list))
]

# Helper function to process frames
def process_frame(frame):
    results = model.predict(source=[frame], conf=0.45, save=False)
    detections = results[0]

    if len(detections) > 0:
        for box in detections.boxes:
            bb = box.xyxy.numpy()[0]
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                2,
            )

            # Display class name and confidence
            label = f"{class_list[clsID]}: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    return frame

# Streamlit App Layout
st.title("**Real-Time Object Tracking**")
st.markdown(
    """
    **Description:**  
    VisionDetect is a real-time object detection tool powered by YOLO. It lets you analyze live webcam feeds or uploaded images/videos to detect objects with state-of-the-art YOLO models.  
    """
)
st.write("---")

# Sidebar for input selection
input_mode = st.sidebar.radio("Choose Input Mode:", ["Webcam", "Upload File"])

if input_mode == "Webcam":
    start_webcam = st.button("🎥 Start Webcam")

    if start_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam.")
        else:
            stframe = st.empty()
            stop_webcam = st.button("🛑 Stop Webcam")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    break

                # Process and display the frame
                processed_frame = process_frame(frame)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_column_width=True)

                # Stop the webcam if the stop button is pressed
                if stop_webcam:
                    break

            cap.release()

elif input_mode == "Upload File":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            # Process uploaded image
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame = process_frame(frame)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, caption="Detected Objects", use_column_width=True)

        elif uploaded_file.name.lower().endswith((".mp4", ".avi")):
            # Process uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_frame(frame)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_column_width=True)

            cap.release()
st.write("---")
