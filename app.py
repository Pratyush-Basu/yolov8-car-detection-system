import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load trained model
model = YOLO("best_new.pt")   # keep model in same folder

st.title("Car Detection & Localization Demo")

CONF = 0.45   # use your tuned threshold

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    image = image.convert("RGB")     
    img_np = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    results = model(img_np, conf=CONF)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.image(img_np, caption="Detected Cars", use_column_width=True)
