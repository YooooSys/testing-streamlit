import cv2

def adjust_bounding_box(x, y, w, h, padding):
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    return x, y, w, h

def draw_bounding_box(image, bbox, label):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image

def add_padding(digit, padding):
    padded_digit = cv2.copyMakeBorder(digit, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    return padded_digit
