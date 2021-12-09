from cv2 import rectangle
from matplotlib.patches import Circle
import numpy as np

from src.face_detection.face_detection import get_circle_bounding_box_for_image

def get_bounding_coords(image, circle):
    x1 = (circle[0][0] - circle[1])
    x2 = (circle[0][0] + circle[1])
    y1 = (circle[0][1] - circle[1])
    y2 = (circle[0][1] + circle[1])
    return (y1,y2, x1,x2)

def extract_coords_features(image, circle , n=400): 
    y1, y2, x1, x2 = get_bounding_coords(image, circle)
    xs = np.random.randint(x1,x2, size=n)
    ys = np.random.randint(y1,y2, size=n)    
    return np.array(list(zip(xs,ys)))

def extract_coords_features_from_rectangle(rectangle, amount_extracted_features=400):
    x, y, w, h = rectangle
    xs = np.random.randint(x, x+w, size=amount_extracted_features)
    ys = np.random.randint(y, y+h, size=amount_extracted_features)    
    return np.array(list(zip(xs, ys)))
  
def align_mean_shape(mean_shape, path):
    circle            = get_circle_bounding_box_for_image(path, 'default')
    center_bb         = circle[0]
    radius            = circle[1]
    scale_factor      = radius/450
    y_constant        = [0, 50*scale_factor]
    mean_shape        = mean_shape * scale_factor
    center_mean_shape = np.mean(mean_shape, axis=0) - y_constant
    difference        =  center_mean_shape - center_bb
    return mean_shape - difference
