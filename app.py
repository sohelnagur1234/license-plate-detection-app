import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Load YOLOv8 model (update this path if needed)
model = YOLO(r"C:\Users\sohel\runs\detect\yolov8_test_run\weights\best.pt")

# Streamlit UI
st.set_page_config(page_title="License Plate Detection", layout="centered")
st.title("🚘 License Plate Detection - IML Project")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display input image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the image temporarily
    image_path = "temp_input.jpg"
    image.save(image_path)

    with st.spinner("Detecting license plate..."):
        results = model.predict(
            source=image_path,
            save=True,
            conf=0.25,
            project="runs/detect",
            name="streamlit_output",
            exist_ok=True
        )

    # Show result
    result_img_path = os.path.join("runs", "detect", "streamlit_output", "temp_input.jpg")
    if os.path.exists(result_img_path):
        st.image(result_img_path, caption="🔍 Detected License Plate", use_column_width=True)
        st.success("Detection complete! ✅")
    else:
        st.error("Something went wrong. 😓")
