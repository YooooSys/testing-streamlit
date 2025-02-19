
import os
import streamlit as st

model_path = os.path.abspath("Model_v4-1.keras")
if os.path.exists(model_path):
    st.error(f"Error: Model file not found at {model_path}. Please upload the file or check the path.")
    
from PIL import Image
from Prediction import preprocess_image, detect_pic
import numpy as np
# Title of the app
st.title("Digit Recognizer")


# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image_arr = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode image using OpenCV in grayscale mode
    preprocessed_image = preprocess_image(image_arr, max_size=1200)

    result = detect_pic(preprocessed_image)

    image = Image.open(result)
    
    # Display the image
    st.image(image, caption='Result', use_container_width=True)
    
    # Optionally, you can save the image to a file
    # image.save("uploaded_image.png")
    
    st.write("Image uploaded successfully!")
else:
    st.write("Please upload an image file.")