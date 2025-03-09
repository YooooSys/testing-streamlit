import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from image_processing import contrast_up
import path_holder
from bounding_box import draw_bounding_box, adjust_bounding_box, add_padding

# Load mô hình đã huấn luyện
model_path = os.path.abspath(path_holder.pre_trained_model_path)

model = tf.keras.models.load_model(model_path)

def detection(image):
    image_ = contrast_up(image)
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_out = cv2.cvtColor(image_, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        #Lọc nhiễu
        tile = w / float(h)
        if (w * h < 100) or (tile < 0.2) or (tile > 1.3):
            continue
        
        #Padding
        padding = 7
        x, y, w, h = adjust_bounding_box(x, y, w, h, padding)
        
        #Đảm bảo bounding box không vượt quá kích thước ảnh
        bbox = (max(0, x), max(0, y), min(w, image.shape[1] - x), min(h, image.shape[0] - y)) #(x, y, w, h)
        #Cắt ảnh chữ số
        digit = binary[y:y+h, x:x+w]
        
        #Thêm padding cho ảnh
        digit = add_padding(digit, padding)
        
        #Resize
        digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit_resized = digit_resized.astype('float32') / 255
        digit_resized = np.expand_dims(digit_resized, axis=-1)
        digit_resized = np.expand_dims(digit_resized, axis=0)
        
        #Áp mô hình
        prediction = model.predict(digit_resized)
        digit_label = np.argmax(prediction)

        # plt.imshow(digit_resized.reshape(28, 28), cmap='gray')
        # plt.title(f'Dự đoán: {digit_label}')
        # plt.show()

        #Vẽ bounding box và kết quả lên ảnh
        image_out = draw_bounding_box(image_out, bbox, digit_label)

    return image_out