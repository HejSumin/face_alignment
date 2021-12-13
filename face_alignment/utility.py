import numpy as np 
import cv2 as cv2
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from face_alignment.alignment import *
from face_detection.face_detection import *
#data = '~/CS-ITU/3-semester/Advanced Machine Learning/Project/data/'
data = '../data/'
annotations = '../data/annotation/'


def get_all_file_names(folder):
    return os.listdir(data+folder)


def get_image(id_image):
    image = cv2.imread(data +"train_1/"+id_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray 

def create_training_data():
    triplets = []
    #NOTE remember to set the correct path
    files = get_all_file_names("train_1")
    #NOTE this is a hyper parameter
    R = 1
    shapes = []
    for f in files:
        I_path                 = f.replace('.jpg', '')
        S_true_x, S_true_y     = get_landmark_coords_from_file(I_path)
        S                      = np.array(list(zip(S_true_x, S_true_y)))
        shapes.append(S)
    mean_shape = center_shape(compute_mean_shape(shapes))
    features = extract_coords_from_mean_shape(mean_shape, offset=50, n=10)
    
    
    for f in files[10:11]:
        I_path     = f.replace('.jpg', '')
        I          = cv2.imread(data +"train_1/"+I_path+".jpg", cv2.IMREAD_GRAYSCALE)
        
        bb         = get_rectangle_bounding_box_for_image(data +"train_1/"+I_path+".jpg", frontalface_config='default')
        
        S_true_x, S_true_y     = get_landmark_coords_from_file(I_path)
        np.random.shuffle(files)
        delta_files = files[:R]

        #NOTE this is the case when delta_file == file
        if I_path in delta_files:
            delta_files = delta_files.remove(I_path)
            delta_files.append(files[20])
        #S_true_x_mean = np.mean(S_true_x)
        #S_true_y_mean = np.mean(S_true_y)

        for d in delta_files:
            S_hat                  = d.replace(".jpg", '')
            S_hat_x, S_hat_y       = get_landmark_coords_from_file(S_hat)

            S_hat_x_mean           = np.mean(S_hat_x)
            S_hat_y_mean           = np.mean(S_hat_y)
            
            bb_center_x            = bb[0][0] + bb[0][2]/2   #bb[0][0] = x coord, bb[0][2] = w
            bb_center_y            = bb[0][1] + 1.2*(bb[0][3]/2)  #bb[0][1]  = y coord, bb[0][3] = h               

            
            diff_x                 = bb_center_x - S_hat_x_mean
            diff_y                 = bb_center_y - S_hat_y_mean

            #NOTE we do this to make a crude approximation of the true shape with the s hat shape to make alignment easier
            
            S_hat_x                += diff_x
            S_hat_y                += diff_y
            S_hat                  = np.array(list(zip(S_hat_x, S_hat_y)))
            
            #NOTE move s hat to origo
            S_hat_mean             = np.mean(S_hat, axis=0)
            S_hat                  = S_hat-S_hat_mean
            
            S_hat_height           = np.max(S_hat[:,1]) - np.min(S_hat[:,1])
            
            # We choose to multiply s hat height by some constant to make up for the extra padding the bounding box adds
            scale_value            = bb[0][3] / (S_hat_height*1.2)
            
            S_hat                  = S_hat *scale_value
            features_hat           = transform_features(mean_shape, S_hat, features)
            
            #NOTE move s hat to center of bb
            S_hat                  = S_hat + [bb_center_x, bb_center_y]
            features_hat           += [bb_center_x, bb_center_y]
            
            
            S_delta_x              = S_true_x - S_hat[:,0]
            S_delta_y              = S_true_y - S_hat[:,1]
            
            S_delta                = np.array(list(zip(S_delta_x, S_delta_y)))
            features_hat           = features_hat.astype(int)
            intensities            = I[np.array(features_hat[:,0]), np.array(features_hat[:,1])]
            
            triplets.append((I, S_hat, S_delta, intensities))

    return np.array(triplets)

def create_test_triplets():
    triplets = []
    
    #NOTE remember to set the correct path
    files = get_all_file_names("train_1")
    
    #NOTE this is a hyper parameter
    R = 20

    for f in files:
        I_path     = f.replace('.jpg', '')
    
        I = cv2.imread(data +"train_1/"+I_path+".jpg")
        
        #return face_detection(I)
        
        S_true_x, S_true_y     = get_landmark_coords_from_file(I_path)
        np.random.shuffle(files)
        delta_files = files[:R]
        if I_path in delta_files:
            delta_files = delta_files.remove(I_path)
            delta_files.append(files[20])
        for d in delta_files:
            S_hat = d.replace(".jpg", '')
            S_hat_x, S_hat_y       = get_landmark_coords_from_file(S_hat)
            S_delta_x              = S_true_x - S_hat_x
            S_delta_y              = S_true_y - S_hat_y
            S_hat                  = np.array(list(zip(S_hat_x, S_hat_y)))
            S_delta                = np.array(list(zip(S_delta_x, S_delta_y)))
        
            triplets.append((I, S_hat, S_delta))
    
    return np.array(triplets)

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


def get_landmark_coords_from_file(id_image):
    id_image = id_image.replace(".jpg", "")
    for i in range(1,2331, 1):
        with open(annotations+str(i)+".txt") as f:
            first_line = f.readline().replace('\n','')
            if (first_line == id_image):
                return read_landmarks_from_file(annotations+str(i)+".txt") 
    return None

def plot_image_given_coords(id_image, coords, colors=['yellow']):
    image = cv2.imread(data +"train_1/"+id_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for c in range(len(coords)):
        coord = list(zip(*coords[c]))
        plt.plot(coord[0],coord[1], color=colors[c], marker='o',  markersize=1,  linestyle = 'None')
    plt.imshow(image)

def compute_mean_shape(images):
    result = np.zeros((194,2))
    for i in range(len(images)):
        shape = images[i]
        result = result + shape
    return result / len(images)
        
        
