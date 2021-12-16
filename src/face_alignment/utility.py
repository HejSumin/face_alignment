import numpy as np
import cv2 as cv2
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import scipy.optimize as opt
import math
from src.face_detection.face_detection import *
import sys


def get_all_file_names(folder):
    return os.listdir(folder)

def get_mean_shape_from_files(filenames, image_to_annotation_dict,annotation_folder_path):

    shapes = []
    for file in filenames:
        I_path                 = image_to_annotation_dict[file.replace('.jpg', '')]
        S_true_x, S_true_y     = read_landmarks_from_file(annotation_folder_path + I_path)
        S                      = np.array(list(zip(S_true_x, S_true_y)))
        shapes.append(S)

    return compute_mean_shape(shapes)

def compute_mean_shape(images):
    result = np.zeros((194,2))
    for i in range(len(images)):
        shape = images[i]
        result = result + shape
    return result / len(images)



def read_landmarks_from_file(file):
    landmarks_x = []
    landmarks_y= []
    f = open(file,"r")
    lines = f.readlines()
    for l in lines[1:]:
        coords = l.replace('\n', '').split(", ")
        landmarks_x.append(float(coords[0]))
        landmarks_y.append(float(coords[1]))
    return (np.array(landmarks_x, dtype=float), np.array(landmarks_y, dtype=float))


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




def center_shape(shape):
    mean = np.mean(shape, axis=0)
    return shape-mean

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

def optimize_equation_8(x_bar, x):
    res  = opt.fmin(func=equation_8, x0=[1,0], args=(x_bar, x), full_output=False, disp=False)
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



def create_training_data(train_folder_path, annotation_folder_path):
    """
    This function prepares the training data for the training of the face alignment algorithm

    It does this by constructing training triplets (septuplets now) from a set of
    training images and corresponding annotations.

        Parameters:
            train_folder_path: The relative position of the folder containing the image filenames
            annotation_folder_path: The relative position of the folder

        Returns:
            training_data


    """
    training_data = []
    image_files = get_all_file_names(train_folder_path)

    annotation_files = get_all_file_names(annotation_folder_path)

    image_to_annotation_dict = {}
    for file in annotation_files:
        with open(annotation_folder_path+file) as f:
            first_line = f.readline().replace('\n','')
        image_to_annotation_dict[first_line] = file


    #NOTE remember to set R
    R = 20

    #calculate mean shape from all shape files
    mean_shape = get_mean_shape_from_files(image_files,image_to_annotation_dict,annotation_folder_path)
    #Center shape around origo to define features in this coordinate system
    mean_shape = center_shape(mean_shape)

    #NOTE remember to set n, which is number of features. Default=400
    features = extract_coords_from_mean_shape(mean_shape, offset=30, n=400)

    for file in tqdm(image_files):
        I_path     = file.replace('.jpg', '')
        I           = cv2.imread(train_folder_path+file, cv2.IMREAD_GRAYSCALE)
        bb         = get_rectangle_bounding_box_for_image(train_folder_path+file, frontalface_config='default')
        if(bb is None):
            continue

        S_true_x, S_true_y     = read_landmarks_from_file(annotation_folder_path + image_to_annotation_dict[file.replace('.jpg', '')])
        S_true                 = np.array(list(zip(S_true_x, S_true_y)))
        np.random.shuffle(image_files)

        #Select the R number of duplicates for image
        delta_files = image_files[:R]

        #NOTE this is the case when delta_file == file
        if I_path in delta_files:
            delta_files = delta_files.remove(I_path)
            delta_files.append(image_files[20])

        for d in delta_files:

            #NOTE Extract landmarks for s hat
            S_hat_image_id         = d.replace(".jpg", '')
            S_hat_path             = annotation_folder_path + image_to_annotation_dict[S_hat_image_id]
            S_hat_x, S_hat_y       = read_landmarks_from_file(S_hat_path)
            S_hat                  = np.array(list(zip(S_hat_x, S_hat_y)))

            #NOTE move s hat to origo
            S_hat                  = center_shape(S_hat)

            #NOTE scalling to bb; We choose to multiply s hat height by some constant to make up for the extra padding the bounding box adds
            S_hat_height           = np.max(S_hat[:,1]) - np.min(S_hat[:,1])
            scale_value            = bb[3] / (S_hat_height*1.3)
            S_hat                  = S_hat *scale_value

            #NOTE warping; we transform from mean shape coordinate system to s hat system
            features_hat           = transform_features(mean_shape, S_hat, features)

            #NOTE Calculate center of bounding box
            bb_center_x            = bb[0] + bb[2]/2   #bb[0][0] = x coord, bb[0][2] = w
            bb_center_y            = bb[1] + 1.1*(bb[3]/2)  #bb[0][1]  = y coord, bb[0][3] = h

            #NOTE move scaled s hat and its features to center of bb
            S_hat                  = S_hat + [bb_center_x, bb_center_y]
            features_hat           += [bb_center_x, bb_center_y]

            #NOTE calculate delta values based scaled and translated s hat and the true shape
            S_delta_x              = S_true_x - S_hat[:,0]
            S_delta_y              = S_true_y - S_hat[:,1]
            S_delta                = np.array(list(zip(S_delta_x, S_delta_y)))

            #NOTE we get the intensities from the images based on the feature points
            features_hat           = features_hat.astype(int)

            try:
                intensities            = I[np.array(features_hat[:,1]), np.array(features_hat[:,0])]
            except:
                continue
            #NOTE we return Image, s hat, s delta, feature intensities values, feature points, and bounding box
            training_data.append((I, S_hat, S_delta, intensities, features_hat, bb, S_true))


    return np.array(training_data)

def preapare_training_data(training_data):
    N = training_data.shape[0]
    I_grayscale_matrix = np.empty((N, _AMOUNT_EXTRACTED_FEATURES))
    S_hat_matrix = np.empty((N,_AMOUNT_LANDMARKS*2))
    S_delta_matrix = np.empty((N,_AMOUNT_LANDMARKS*2))
    S_true_matrix = np.empty((N, _AMOUNT_LANDMARKS*2))

    for i in range(0, training_data.shape[0]):
        S_delta = training_data[i,2].flatten().reshape(388,1).T
        S_hat = training_data[i,1].flatten().reshape(388,1).T
        I_grayscale = training_data[i,3]
        S_true = training_data[i,6]
        I_grayscale_matrix[i] = I_grayscale
        S_hat_matrix[i] = S_hat
        S_delta_matrix[i] = S_delta
        S_true_matrix[i] = S_true

    return (I_grayscale_matrix, S_hat_matrix, S_delta_matrix, S_true_matrix)
