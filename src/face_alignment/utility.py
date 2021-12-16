from src.face_alignment.alignment import *
from src.face_detection.face_detection import *

import numpy as np 
import cv2 as cv2
import matplotlib.pyplot as plt
import os

# Paths
data_path = 'data/'
annotations_path = 'data/annotation/'

# Hyperparameter
_R = 20
_AMOUNT_EXTRACTED_FEATURES = 400
_AMOUNT_LANDMARKS = 194

def create_training_triplets(train_images_path):
    image_file_names = _get_all_file_names_for_folder(train_images_path)
    N = len(image_file_names)*_R
    I_grayscale_matrix = np.empty((N, _AMOUNT_EXTRACTED_FEATURES))
    S_hat_matrix = np.empty((N,_AMOUNT_LANDMARKS*2))
    S_delta_matrix = np.empty((N,_AMOUNT_LANDMARKS*2))
    S_true_matrix = np.empty((N, _AMOUNT_LANDMARKS*2))

    for i, image_file_name in enumerate(image_file_names):
        image_path = train_images_path+image_file_name
        I_grayscale = _extract_features_for_image(image_path)
        if I_grayscale is None: # skip image if bounding box is not found (None)
            continue

        image_id = image_file_name.replace('.jpg', '')
        S_true = get_landmark_coords_for_image(image_id)

        np.random.shuffle(image_file_names)
        delta_image_file_names = image_file_names[:_R]
        if image_file_name in delta_image_file_names:
            delta_image_file_names.remove(image_file_name)
            delta_image_file_names.append(image_file_names[_R])

        for r, delta_image_file_name in enumerate(delta_image_file_names):
            delta_image_id = delta_image_file_name.replace('.jpg', '')
            S_hat = get_landmark_coords_for_image(delta_image_id)
            S_delta = S_true - S_hat

            index = (i*_R)+r
            I_grayscale_matrix[index] = I_grayscale
            S_hat_matrix[index] = S_hat
            S_delta_matrix[index] = S_delta
            S_true_matrix[index] = S_true

    return (I_grayscale_matrix, S_hat_matrix, S_delta_matrix, S_true_matrix)

def prepare_training_data(training_data):
    N = training_data.shape[0]
    I_grayscale_matrix = np.empty((N, _AMOUNT_EXTRACTED_FEATURES))
    S_hat_matrix = np.empty((N,_AMOUNT_LANDMARKS*2))
    S_delta_matrix = np.empty((N,_AMOUNT_LANDMARKS*2))
    S_true_matrix = np.empty((N, _AMOUNT_LANDMARKS*2))

    for i in range(0, training_data.shape[0]): # [x1,y1,x2,y2..]
        S_delta = training_data[i,2].flatten().reshape(388,1).T # (1, 388)
        S_hat = training_data[i,1].flatten().reshape(388,1).T
        I_grayscale = training_data[i,3]
        S_true = training_data[i,6]
        I_grayscale_matrix[i] = I_grayscale
        S_hat_matrix[i] = S_hat
        S_delta_matrix[i] = S_delta
        S_true_matrix[i] = S_true

    return (I_grayscale_matrix, S_hat_matrix, S_delta_matrix, S_true_matrix)

def _extract_features_for_image(image_path):
    rectangle_bounding_box = get_rectangle_bounding_box_for_image(image_path)
    if rectangle_bounding_box is None: # check if bounding box was found in image
       return None 
    else:
        extracted_coords_features = extract_coords_features_from_rectangle(rectangle_bounding_box, _AMOUNT_EXTRACTED_FEATURES)

        I_grayscale_all_features = _get_image(image_path)
        I_grayscale = []
        for index in range(0, _AMOUNT_EXTRACTED_FEATURES):
            x, y = extracted_coords_features[index]
            I_grayscale.append(I_grayscale_all_features[y, x])
        return I_grayscale

def _get_all_file_names_for_folder(folder_path):
    return os.listdir(folder_path)

def _get_image(image_file_path):
    image = cv2.imread(image_file_path)
    I_grayscale_full_features = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return I_grayscale_full_features 

def get_landmarks_from_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()[1:]
        landmarks = []
        for i, line in enumerate(lines):
            coords = [float(x) for x in line.replace('\n', '').split(' , ')]      
            landmarks.append(coords[0])
            landmarks.append(coords[1])
        return np.array(landmarks, ndmin=2)

def get_landmark_coords_for_image(image_id):
    for annotation_file_name in os.listdir(annotations_path):
        annotation_file_path = annotations_path + annotation_file_name
        with open(annotation_file_path) as annotation_file:
            annotation_file_image_id = annotation_file.readline().replace('\n','')
            if annotation_file_image_id == image_id:
                return get_landmarks_from_file(annotation_file_path) 
    return None

def plot_image_given_landmarks(image_file_path, landmarks, colors=['yellow']):
    image = cv2.imread(image_file_path)
    I_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(0, len(landmarks-1), 2):
        landmark_x = landmarks[:,i]
        landmark_y = landmarks[:,i+1]
        plt.plot(landmark_x,landmark_y, color=colors[i], marker='o',  markersize=1,  linestyle = 'None')
    plt.imshow(I_grayscale)

def compute_mean_shape(S_true_matrix):
    return np.mean(S_true_matrix, axis=0)
    