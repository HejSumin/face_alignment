from matplotlib.patches import Circle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import scipy.optimize as opt
import math
from face_detection.face_detection import get_circle_bounding_box_for_image


def rotate_matrix(theta):
    T = np.zeros((2,2))
    T[0][0] = math.cos(theta) 
    T[0][1] = -math.sin(theta)
    T[1][0] = math.sin(theta)
    T[1][1] = math.cos(theta)
    return T

def equation_8(params, x_bar, x):
    scale = params[0]
    theta = params[1]
    R     = rotate_matrix(theta) 
    return np.sum(euclidean_distances(x_bar - np.dot(x*scale, R)))


def center_shape(shape):
    mean = np.mean(shape, axis=0)
    return shape-mean


    
def optimize_equation_8(x_bar, x):
    res  = opt.fmin(func=equation_8, x0=[0,0], args=(x_bar, x))
    return res


def extract_coords_from_mean_shape(mean_shape, offset=100, n=400):
    
    x_mean_shape =  [x[0] for x in mean_shape]
    y_mean_shape =  [x[1] for x in mean_shape]

    xmin = np.min(x_mean_shape) - offset
    xmax = np.max(x_mean_shape) + offset
    ymin = np.min(y_mean_shape) - offset * 1.5
    ymax = np.max(y_mean_shape) + offset

    xs = np.random.randint(xmin,xmax, size=n)
    ys = np.random.randint(ymin,ymax, size=n)    
    return np.array(list(zip(xs,ys)))
    
def get_bounding_coords( circle):
    x1 = (circle[0][0][0] - circle[0][1])
    x2 = (circle[0][0][0] + circle[0][1])
    y1 = (circle[0][0][1] - circle[0][1])
    y2 = (circle[0][0][1] + circle[0][1])
    return (y1,y2, x1,x2)
            


def extract_coords_features(circle , n=400): 
    y1, y2, x1, x2 = get_bounding_coords(image, circle)
    xs = np.random.randint(x1,x2, size=n)
    ys = np.random.randint(y1,y2, size=n)    
    return np.array(list(zip(xs,ys)))

    
  
def align_mean_shape(mean_shape, path):
    circle            = get_circle_bounding_box_for_image(path, 'default')[0]
    center_bb         = circle[0]
    radius            = circle[1]
    scale_factor      = radius/450
    y_constant        = [0, 50*scale_factor]
    mean_shape        = mean_shape * scale_factor
    center_mean_shape = np.mean(mean_shape, axis=0) - y_constant
    difference        =  center_mean_shape - center_bb
    return mean_shape - difference
  
