from matplotlib.patches import Circle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import scipy.optimize as opt
import math
import sys
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
    res  = opt.fmin(func=equation_8, x0=[1,0], args=(x_bar, x))
    return res


def find_closest_landmark(feature, landmarks):
    
    min_distance = sys.float_info.max
    closest_landmark = -1

    for index, landmark in enumerate(landmarks): 
            distance = np.linalg.norm( landmark - feature)

            if distance < min_distance:
                min_distance = distance 
                closest_landmark = index
                
    return closest_landmark

def gen_list_of_closest_landmarks(features, landmarks):
    closest_landmarks = []

    for feature in features:
        closest_landmarks.append(find_closest_landmark(feature, landmarks))
        
    return closest_landmarks
    

def transform_features(s0, s1, features0):
    #Takes a set of landmarks s0 with a corresponding set of features features0 and warps these features to fit on
    #the set of landmarks given by s1. 
    
    #Assumption: Both sets of landmarks s0 and s1 are centered when given as input
    
    opt = optimize_equation_8(s1, s0)
    R = rotate_matrix(opt[1])
    scale = opt[0]
    
    closest_landmarks = gen_list_of_closest_landmarks(features0, s0)
    landmarks_positions = np.array([list(s0[i]) for i in closest_landmarks ])
    offsets = features0 - landmarks_positions
    
    landmarks_positions_s1 = np.array([list(s1[i]) for i in closest_landmarks ])
    features1 = landmarks_positions_s1 + np.dot(offsets*scale, R)
    
    return features1 
    


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
  
