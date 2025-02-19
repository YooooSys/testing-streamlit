import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình đã huấn luyện
model_path = os.path.abspath("Model_v4-1.keras")

model = tf.keras.models.load_model(model_path)

def add_padding(digit, padding):
    padded_digit = cv2.copyMakeBorder(digit, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    return padded_digit

def adjust_bounding_box(x, y, w, h, padding=7):
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    return x, y, w, h

def detect_pic(image_):
    image = contrast_up(image_)

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
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
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
        cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_out, str(digit_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return image_out
    
def preprocess_image(image, max_size):
    # Đọc ảnh
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Không thể đọc ảnh.")
        return None

    # Lấy kích thước ảnh
    height, width = image.shape

    # Tính tỉ lệ resize
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image

# Đường dẫn đến ảnh cần nhận diện
image_path = 'image/bigtest/big_test.jpg'

# Tiền xử lý ảnh
def contrast_up(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

