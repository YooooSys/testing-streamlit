import streamlit as st
from PIL import Image

# Title of the app
st.title("Image Uploader")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Optionally, you can save the image to a file
    # image.save("uploaded_image.png")
    
    st.write("Image uploaded successfully!")
else:
    st.write("Please upload an image file.")