import cv2

def adjust_bounding_box(x, y, w, h, padding):
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    return x, y, w, h

def draw_bounding_box(image, bbox, label):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Draw the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Place the text inside the bounding box
    # Position the text at (x + 5, y + 25) to offset it slightly from the top-left corner
    text_position = (x + 5, y + 25)
    cv2.putText(image, str(label), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image

def add_padding(digit, padding):
    padded_digit = cv2.copyMakeBorder(digit, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    return padded_digit
