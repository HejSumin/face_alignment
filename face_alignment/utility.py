import numpy as np 
import cv2 as cv2
import matplotlib.pyplot as plt
import pandas as pd
import os, sys

#data = '~/CS-ITU/3-semester/Advanced Machine Learning/Project/data/'
data = '../data/'
annotations = '../data/annotation/'


def get_all_file_names(folder):
    return os.listdir(data+folder)


def get_image(id_image):
    image = cv2.imread(data +"train_1/"+id_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray 

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
        shape = images[i][1]
        result = result + shape
    return result / len(images)
        
        
