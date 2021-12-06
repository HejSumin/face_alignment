from time import time
import timeit

# TODO move to seperate package
from tree_fitting import *

_LEARNING_RATE = 0.1
_K = 500

# I_image_matrix [n, 194]
# I_grayscale_triplets_matrix [N, 194]
def build_regression_trees(I_image_matrix, I_grayscale_triplet_matrix, S_delta_matrix, f_0_matrix):
    f_k_minus_1_matrix = f_0_matrix

    for k in range(0, _K):
        r_i_k_matrix = generate_residual_image_vector_matrix(S_delta_matrix, f_k_minus_1_matrix)
        regression_tree = generate_regression_tree(I_grayscale_triplet_matrix, r_i_k_matrix)
        f_k_matrix = update_f_k_matrix(regression_tree, f_k_minus_1_matrix, I_image_matrix)
        f_k_minus_1_matrix = f_k_matrix

    return f_k_minus_1_matrix

# TODO align shapes of f_k matrices, etc.

def generate_residual_image_vector_matrix(S_delta_matrix, f_k_minus_1_matrix): #[N, 194], [n, 194]
    R = S_delta_matrix.shape[0] / f_k_minus_1_matrix.shape[0]
    f_k_minus_1_triplet_matrix = np.repeat(f_k_minus_1_matrix, repeats=R, axis=0)
    return S_delta_matrix - f_k_minus_1_triplet_matrix

def update_f_k_matrix(regression_tree, f_k_minus_1_matrix, I_image_matrix, learning_rate=_LEARNING_RATE):
    g_k_matrix = np.empty_like(f_k_minus_1_matrix)
    for index, _ in enumerate(I_image_matrix):
        g_k_matrix[index] = get_avarage_residual_image_vector(regression_tree, I_image_matrix[0])
    
    f_k_matrix = f_k_minus_1_matrix + learning_rate * g_k_matrix
    return f_k_matrix

# [0,0,[grayscale values],0,1,1,1,1,2,2,2,2,]
images = 5

I_image_matrix = np.random.randint(0, 256, (images, 400))
I_grayscale_triplets_matrix = np.repeat(I_image_matrix, repeats=20, axis=0) # shape (N=n*R, #extraced pixels) # TODO Rename matrix to something with triplets
print(I_grayscale_triplets_matrix.shape)
residual_image_vector_matrix = np.random.rand(20*images, 194) # only positive values for test example ; shape (N=n*R, 194)
S_delta_triplets_matrix = np.random.randint(1, 10, (20*images, 194)) * np.random.rand(20*images, 194) # 20 = R , images = amount of actual Images I
f_0_matrix = np.random.randint(1, 10, (images, 194)) * np.random.rand(images, 194)

from datetime import datetime

now = datetime.now()
r_t_matrix = build_regression_trees(I_image_matrix, I_grayscale_triplets_matrix, S_delta_triplets_matrix, f_0_matrix)
#print(r_t_matrix)
print(datetime.now() - now)

# TODO rename I_image_matrix stuff
# TODO connect to mean shape initializing step
# TODO remove random matrix calculation