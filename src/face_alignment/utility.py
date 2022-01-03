import numpy as np
import cv2 as cv2
import os
from sklearn.utils.validation import has_fit_parameter
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import scipy.optimize as opt
import math
from src.face_detection.face_detection import *
import sys
from numba import jit

"""
Hyperparameters

"""

_R = 20

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

@jit(nopython=True)
def find_closest_landmark(feature, landmarks):

    min_distance = 10000000
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

# Compute average landmark distance from the ground truth landamarks normalized by the distance between eyes for a single image.
# TODO compute_mean_error function needs to be implemented to handle multiple images
def compute_error(shape, S_true):
    interocular_distance = np.linalg.norm(S_true[153]-S_true[114])
    average_distance = np.linalg.norm(shape - S_true, axis=1)/interocular_distance
    return average_distance.mean()

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
def create_training_data(train_folder_path, annotation_folder_path):
    training_data = []
    image_files = get_all_file_names(train_folder_path)

    bb_target_size = 500 #TODO Change that?

    image_to_annotation_dict = build_image_to_annotation_dict(annotation_folder_path)

    # calculate mean shape (S_mean) from all shape files
    S_mean = get_mean_shape_from_files(image_files, image_to_annotation_dict, annotation_folder_path)
    # center mean shape around the origin to define features in this coordinate system
    S_mean_centered = center_shape(S_mean)
    np.save("np_data/S_mean_centered", S_mean_centered)

    #NOTE remember to set n, which is number of features. Default=400 #TODO make this n a parameter of the function!
    features_mean = extract_coords_from_mean_shape(S_mean_centered, offset=20, n=400)
    np.save("np_data/features_mean", features_mean)

    for I_file_name in tqdm(image_files):
        prepare_result = prepare_image_and_bounding_box(train_folder_path+I_file_name, bb_target_size)
        if prepare_result is None:
            continue
        I_resized, bb_scaled, bb_scale_factor = prepare_result

        I_padded, h_pad, w_pad = pad_image_with_zeros(I_resized)
       
        #NOTE we use the the scale and padding values to move the true shape to the new image
        I_id = I_file_name.replace('.jpg', '')
        S_true = scale_S_true_to_bb_and_pad(I_id, annotation_folder_path, image_to_annotation_dict, bb_scale_factor, w_pad, h_pad)
        
        np.random.shuffle(image_files)
        # select the R number of duplicates for image
        delta_files = image_files[:_R]

        #NOTE this is the case when delta_file == file
        if I_file_name in delta_files:
            delta_files.remove(I_file_name)
            delta_files.append(image_files[_R])

        for delta_file_name in delta_files:
        
            #NOTE extract landmarks for S_hat
            S_hat_image_id = delta_file_name.replace(".jpg", '')
            S_hat_x, S_hat_y = read_landmarks_from_file(annotation_folder_path + image_to_annotation_dict[S_hat_image_id])
            S_hat_x += w_pad
            S_hat_y += h_pad
            S_hat_raw = np.array(list(zip(S_hat_x, S_hat_y)), dtype=np.uint16)

            #NOTE move s hat to origo
            S_hat_centered = center_shape(S_hat_raw)
            S_hat, features_hat = prepare_S_hat_and_features_hat(S_hat_centered, S_mean_centered, features_mean, bb_scaled, w_pad, h_pad)

            #NOTE calculate delta values based scaled and translated s hat and the true shape
            S_delta_x = S_true[:, 0] - S_hat[:, 0]
            S_delta_y = S_true[:, 1] - S_hat[:, 1]
            S_delta = np.array(list(zip(S_delta_x, S_delta_y)), np.float32)

            try:
                intensities = I_padded[np.array(features_hat[:,1]), np.array(features_hat[:,0])]
            except:
                continue
            #NOTE we return Image, s hat, s delta, feature intensities values, feature points, and bounding box
            training_data.append((I_padded, S_hat, S_delta, intensities, features_hat, bb_scaled, S_true))

    return np.array(training_data, dtype=object)

def build_image_to_annotation_dict(annotation_folder_path):
    annotation_files = get_all_file_names(annotation_folder_path)

    image_to_annotation_dict = {}
    for annotation_file in annotation_files:
        with open(annotation_folder_path+annotation_file) as f:
            first_line = f.readline().replace('\n','')
        image_to_annotation_dict[first_line] = annotation_file

    return image_to_annotation_dict

def prepare_S_hat_and_features_hat(S_hat_centered, S_mean_centered, features_mean, bb_scaled, w_pad, h_pad):
    bb_x, bb_y, bb_w, bb_h = bb_scaled[0], bb_scaled[1], bb_scaled[2], bb_scaled[3]
    S_hat_scaled = scale_S_hat_to_bb(S_hat_centered, bb_h)

    #NOTE warping; we transform from mean shape coordinate system to S_hat system
    features_hat_transformed = transform_features(S_mean_centered, S_hat_scaled, features_mean)
    #NOTE Calculate center of bounding box
    bb_center_x            = (bb_x + bb_w/2) + w_pad  
    bb_center_y            = (bb_y + 1.1*(bb_h/2)) + h_pad #TODO constant should be a parameter

    #NOTE move scaled S_hat and its features to center of bb
    S_hat                  = S_hat_scaled + [bb_center_x, bb_center_y]
    features_hat           = features_hat_transformed + [bb_center_x, bb_center_y]

    return S_hat, features_hat.astype(np.uint16) #TODO uint8?

def scale_S_true_to_bb_and_pad(I_id, annotation_folder_path, image_to_annotation_dict, bb_scale_factor, w_pad, h_pad):
    S_true_x, S_true_y = read_landmarks_from_file(annotation_folder_path + image_to_annotation_dict[I_id])
    S_true_x = S_true_x * bb_scale_factor
    S_true_y = S_true_y * bb_scale_factor
    S_true_x += w_pad
    S_true_y += h_pad
    S_true = np.array(list(zip(S_true_x, S_true_y)), dtype=np.uint16)
    return S_true

def scale_S_hat_to_bb(S_hat_centered, bb_height):
    #NOTE scalling to bb; We choose to multiply s hat height by some constant to make up for the extra padding the bounding box adds
    S_hat_height = np.max(S_hat_centered[:,1]) - np.min(S_hat_centered[:,1])
    scale_factor = bb_height / (S_hat_height*1.3) #TODO make 1.3 a parameter
    S_hat_scaled = S_hat_centered * scale_factor
    return S_hat_scaled

def find_bb_scale_factor(bb, bb_target_size):
    bb_width = bb[2]
    bb_scale_factor = bb_target_size / bb_width
    return bb_scale_factor

def scale_bb(bb, bb_scale_factor):
    bb[2] = bb[2] * bb_scale_factor
    bb[3] = bb[3] * bb_scale_factor
    bb[0] = bb[0] * bb_scale_factor
    bb[1] = bb[1] * bb_scale_factor
    return bb

def resize_image(I, I_width, I_height, bb_scale_factor):
    return cv2.resize(I, (int(I_width*bb_scale_factor), int(I_height*bb_scale_factor)), interpolation=cv2.INTER_LINEAR)

def pad_image_with_zeros(I_resized):
    I_resized_height, I_resized_width = I_resized.shape # TODO is that change correct? Getting the resized image shape to apply the padding!   
    #NOTE padding the image with zeros in order to avoid index out of bound errors
    h_pad = (int((I_resized_height / 100) * 20)) # TODO extract hyperparameters
    w_pad = (int((I_resized_width / 100) * 20))
    I_padded = cv2.copyMakeBorder(I_resized, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT)
    return I_padded, h_pad, w_pad

def prepare_image_and_bounding_box(I_file_path, bb_target_size):
    I_raw = cv2.imread(I_file_path, cv2.IMREAD_GRAYSCALE)
    I_raw_height, I_raw_width = I_raw.shape
    bb_raw = get_rectangle_bounding_box_for_image(I_file_path, frontalface_config='default')

    if(bb_raw is None):
        return None
    else:      
        bb_scale_factor = find_bb_scale_factor(bb_raw, bb_target_size)
        bb_scaled = scale_bb(bb_raw, bb_scale_factor)
        I_resized = resize_image(I_raw, I_raw_width, I_raw_height, bb_scale_factor)

        return I_resized, bb_scaled, bb_scale_factor       

def prepare_training_data_for_tree_cascade(training_data):
    N = training_data.shape[0]
    amount_extracted_features = training_data[0, 3].shape[0]
    amount_landmarks = training_data[0, 1].shape[0]

    I_intensities_matrix = np.empty((N, amount_extracted_features), dtype=np.int16)
    features_hat_matrix = np.empty((N, amount_extracted_features*2), dtype=np.int16)
    S_hat_matrix = np.empty((N, amount_landmarks*2), dtype=np.uint16)
    S_delta_matrix = np.empty((N, amount_landmarks*2), dtype=np.float32)
    S_true_matrix = np.empty((N, amount_landmarks*2), dtype=np.uint16)

    for i in range(0, training_data.shape[0]):
        I_intensities = training_data[i, 3]
        features_hat = training_data[i, 4].flatten().reshape(amount_extracted_features*2, 1).T
        S_hat = training_data[i, 1].flatten().reshape(amount_landmarks*2, 1).T
        S_delta = training_data[i, 2].flatten().reshape(amount_landmarks*2, 1).T
        S_true = training_data[i, 6].flatten().reshape(amount_landmarks*2, 1).T

        I_intensities_matrix[i] = I_intensities
        features_hat_matrix[i] = features_hat
        S_hat_matrix[i] = S_hat
        S_delta_matrix[i] = S_delta
        S_true_matrix[i] = S_true

    return (I_intensities_matrix, np.array(features_hat_matrix, dtype=np.uint16), S_hat_matrix, S_delta_matrix, S_true_matrix)

def transformation_between_cascades(S_0, S_new, features_0):
     # calculate mean of S_0 to move it (the shape) and its features to the origin
     S_0_mean = np.mean(S_0, axis=0)
     S_0_centered = S_0 - S_0_mean
     features_0_centered = features_0 - S_0_mean

     # calculate mean of S_new to move it (the shape) and its features to the origin

     S_new_mean = np.mean(S_new, axis=0)
     S_new_centered = S_new - S_new_mean

     features_new = transform_features(S_0_centered, S_new_centered, features_0_centered).astype(int)

     # move features_new back to the image's coordinate system
     features_new += S_new_mean.astype(int)

     return features_new

def update_training_data_with_tree_cascade_result(all_S_0, all_features_0, S_hat_matrix_new, S_delta_matrix_new, training_data,last_run):

    N = training_data.shape[0]
    amount_extracted_features = training_data[0, 3].shape[0]
    amount_landmarks = training_data[0, 1].shape[0]

    x_mask = [x for x in range(0, amount_landmarks*2-1, 2)]
    y_mask = [x for x in range(1, amount_landmarks*2, 2)]

    I_intensities_matrix_new = np.empty((N, amount_extracted_features), dtype=np.int16)

    for i in tqdm(range(0, training_data.shape[0]), desc="update training data"):
        I            = training_data[i, 0]
        features_hat = training_data[i, 4]
        S_hat        = training_data[i, 1]

        S_0          = all_S_0[i]
        features_0   = all_features_0[i]

        S_hat_new    = np.array(list(zip(S_hat_matrix_new[i,x_mask], S_hat_matrix_new[i,y_mask])))
        S_delta_new  = np.array(list(zip(S_delta_matrix_new[i,x_mask], S_delta_matrix_new[i,y_mask])))

        if not last_run:

            features_hat_new = transformation_between_cascades(S_0, S_hat_new, features_0)

            try:
                intensities_new = I[np.array(features_hat_new[:, 1]), np.array(features_hat_new[:, 0])]

            except Exception as e:
                print(e)
                data = np.array([I,features_hat, features_hat_new, S_hat, S_hat_new ], dtype=object)

                np.save("failed_transformations/data"+str(i), data)
                print(i)
                intensities_new = training_data[i, 3]

        else:
            #No need to transform features if last run, so just return old values
            features_hat_new = features_hat
            intensities_new = training_data[i, 3]

        training_data[i, 1] = S_hat_new
        training_data[i, 2] = S_delta_new
        training_data[i, 3] = intensities_new
        training_data[i, 4] = features_hat_new

        I_intensities_matrix_new[i] = intensities_new

    return training_data, I_intensities_matrix_new

#TODO: will be removed after updating it with simpler method
def get_scaled_landmarks(train_folder_path, annotation_folder_path):
    landmarks = []
    image_files = get_all_file_names(train_folder_path)

    bb_target_size = 500 #TODO Change that?

    annotation_files = get_all_file_names(annotation_folder_path)

    image_to_annotation_dict = {}
    for annotation_file in annotation_files:
        with open(annotation_folder_path+annotation_file) as f:
            first_line = f.readline().replace('\n','')
        image_to_annotation_dict[first_line] = annotation_file


    for I_file_name in tqdm(image_files):
        prepare_result = prepare_image_and_bounding_box(train_folder_path+I_file_name, bb_target_size)
        if prepare_result is None:
            continue
        I_resized, bb_scaled, bb_scale_factor = prepare_result

        I_padded, h_pad, w_pad = pad_image_with_zeros(I_resized)
        I_id = I_file_name.replace('.jpg', '')

        #NOTE we use the the scale and padding values to move the true shape to the new image
        S_true_x, S_true_y      = read_landmarks_from_file(annotation_folder_path + image_to_annotation_dict[I_id])
        S_true_x                = S_true_x*bb_scale_factor
        S_true_y                = S_true_y*bb_scale_factor
        S_true_x               += w_pad
        S_true_y               += h_pad
        S_true                 = np.array(list(zip(S_true_x, S_true_y)))
        np.random.shuffle(image_files)

        landmarks.append(S_true)

    return np.array(landmarks, dtype=object)