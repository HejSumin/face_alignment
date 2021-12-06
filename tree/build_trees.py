from timeit import default_timer as timer
from datetime import timedelta
from tree_fitting import *

_LEARNING_RATE = 0.1
_K = 500

def build_regression_trees(I_grayscale_matrix, S_delta_matrix, f_0_matrix):
    f_k_minus_1_matrix = f_0_matrix

    for k in range(0, _K):
        r_i_k_matrix = calculate_residuals_matrix(S_delta_matrix, f_k_minus_1_matrix)
        regression_tree = generate_regression_tree(I_grayscale_matrix, r_i_k_matrix)
        f_k_matrix = update_f_k_matrix(regression_tree, f_k_minus_1_matrix)
        f_k_minus_1_matrix = f_k_matrix

    return f_k_minus_1_matrix

def calculate_residuals_matrix(S_delta_matrix, f_k_minus_1_matrix):
    return S_delta_matrix - f_k_minus_1_matrix

def update_f_k_matrix(regression_tree, f_k_minus_1_matrix, learning_rate=_LEARNING_RATE):
    g_k_matrix = regression_tree.get_avarage_residuals_matrix()
    f_k_matrix = f_k_minus_1_matrix + learning_rate * g_k_matrix
    return f_k_matrix

# [0,0,[grayscale values],0,1,1,1,1,2,2,2,2,]
images = 200

n_image_matrix = np.random.randint(0, 256, (images, 400))
I_grayscale_matrix = np.repeat(n_image_matrix, repeats=20, axis=0) # shape (N=n*R, #extraced pixels)
S_delta_matrix = np.random.randint(1, 10, (20*images, 194)) * np.random.rand(20*images, 194) # 20 = R , images = amount of actual Images I
f_0_matrix = np.random.randint(1, 10, (images*20, 194)) * np.random.rand(images*20, 194)

start = timer()
r_t_matrix = build_regression_trees(I_grayscale_matrix, S_delta_matrix, f_0_matrix)
end = timer()
print(r_t_matrix)
print("Time: ", timedelta(seconds=end-start))

# TODO connect to mean shape initializing step
# TODO remove random matrix calculation