import streamlit as st
from PIL import Image
from Prediction import detection
from image_processing import preprocess_image
import numpy as np
import cv2
from paddleocr import PaddleOCR


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

if st.button("Detect text"):

    if uploaded_file is not None:
        ocr = PaddleOCR(use_angle_cls=True , rec_char_type="en", det_db_score_mode="slow")

        # Loading and resizing image if it's too large
        image = cv2.imread(uploaded_file.read())

        image = image_resizer(image)

        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        # Apply dilation to separate characters
        kernel = np.ones((2,2   ), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        result = ocr.ocr(image, cls=True)

        result = result[0] 

        # get boxes data boxes: [ [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], 
        #                         [......................................], ... ]
        boxes = [line[0] for line in result]
        for box in boxes:
            
            box = np.array([(int(point[0]), int(point[1])) for point in box])
            
            #draw outline from given points info
            cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

        st.image(image, caption='Result', use_container_width=True)

    else:
        st.write("Please upload an image file")