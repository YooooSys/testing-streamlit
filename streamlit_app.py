import streamlit as st
from PIL import Image
from Prediction import detection
from image_processing import preprocess_image
import numpy as np
import cv2


def image_resizer(image, max_width=1920, max_height=1000):
    # shape[0] : height
    # shape[1] : widht 
    if image.shape[0] > 1080:
        image = cv2.resize(image, (int(image.shape[1] / (image.shape[0] / max_height)), max_height))

    elif image.shape[1] > 1920:
        image = cv2.resize(image, (max_width, int(image.shape[0] / (image.shape[1] / max_width))))
    
    return image

# Title
st.title("Digit Recognizer")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Digit Recognition using contour"):

    if uploaded_file is not None:
        st.write("Image uploaded successfully! Now processing...")
        image_arr = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Decode image using OpenCV
        preprocessed_image = preprocess_image(image_arr, max_size=1200)

        # Digit detection
        result = detection(preprocessed_image)
        image_out = Image.fromarray(result)
        
        # Display the image
        st.image(image_out, caption='Result', use_container_width=True)
        
        # image.save("uploaded_image.png")
        
    else:
        st.write("Please upload an image file.")