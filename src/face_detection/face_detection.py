import cv2 as cv2

def _get_bounding_box_for_image(image_path, frontalface_config='default', circle=True):
    frontalface_config_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if frontalface_config =='default' else cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(frontalface_config_path)

    image = cv2.imread(image_path)
    
    width, height, _ = image.shape
    bounding_box_minSize = (width//8, height//8)

    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_grayscale, scaleFactor=1.1, minNeighbors=5, minSize=bounding_box_minSize, flags = cv2.CASCADE_FIND_BIGGEST_OBJECT)

    image_bounding_boxes = []    
    for (x, y, w, h) in faces:
        if circle:
            center = (x+(w//2), y+(h//2))
            radius = int(w//1.5)
            cv2.circle(image, center, radius, (255, 0, 0), 2)
            image_bounding_boxes.append((center, radius))
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            image_bounding_boxes.append([x, y, w, h])

    return image_bounding_boxes 

def get_circle_bounding_box_for_image(image_path, frontalface_config='default'):
    circle_bounding_boxes = _get_bounding_box_for_image(image_path, frontalface_config, circle=True)
    return circle_bounding_boxes[0] if len(circle_bounding_boxes) > 0 else None

def get_rectangle_bounding_box_for_image(image_path, frontalface_config='default'):
    rectangle_bounding_boxes = _get_bounding_box_for_image(image_path, frontalface_config, circle=False)
    return rectangle_bounding_boxes[0] if len(rectangle_bounding_boxes) > 0 else None
