import cv2

def preprocess_image(image, max_size):
    # Đọc ảnh
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Cannot read the image!")
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

# Tiền xử lý ảnh
def contrast_up(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
