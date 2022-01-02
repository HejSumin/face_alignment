from numpy.lib.npyio import save
from src.tree.tree_fitting import *
from src.face_alignment.utility import *
from tqdm import tqdm

_DEBUG = True

"""
Hyperparameters

Parameters
    ----------
    _LEARNING_RATE : how much does the result of each tree influence the overall result
    _K : amount of trees per cascade
    _T : amount of cascades
"""
_LEARNING_RATE = 0.1
_K = 200
_T = 3

def train_multiple_cascades(training_data, model_output_path="run_output/", regression_tree_max_depth=5, use_exponential_prior=True):
    I_intensities_matrix, features_hat_matrix, S_hat_matrix, S_delta_matrix, S_true_matrix = prepare_training_data_for_tree_cascade(training_data)

    for t in tqdm(range(0, _T), desc="T cascades"):
        last_run = t == _T-1

        r_t_matrix, model_regression_trees, f_0_matrix = train_single_cascade(I_intensities_matrix, features_hat_matrix, S_delta_matrix, regression_tree_max_depth, use_exponential_prior)
        save_regression_trees_to_file(model_regression_trees, model_output_path, t, regression_tree_max_depth)
        np.save(model_output_path +"run_output_model_f_0_matrix_" + str(t), f_0_matrix)


        S_hat_matrix = S_hat_matrix + r_t_matrix
        S_delta_matrix = S_true_matrix - S_hat_matrix

        training_data_new, I_intensities_matrix_new = update_training_data_with_tree_cascade_result(S_hat_matrix, S_delta_matrix, training_data, last_run)
        training_data = training_data_new
        np.save("saved_while_training/t_data" + str(t), training_data)
        I_intensities_matrix = I_intensities_matrix_new

    return training_data

def train_single_cascade(I_intensities_matrix, features_hat_matrix, S_delta_matrix, regression_tree_max_depth, use_exponential_prior):
    model_regression_trees = []

    f_0_matrix = calculate_f_0_matrix(S_delta_matrix)

    f_k_minus_1_matrix = f_0_matrix

    for k in tqdm(range(0, _K), desc="K trees"):
        r_i_k_matrix = calculate_residuals_matrix(S_delta_matrix, f_k_minus_1_matrix)

        regression_tree = generate_regression_tree(I_intensities_matrix, r_i_k_matrix, features_hat_matrix, regression_tree_max_depth, use_exponential_prior)
        model_regression_trees.append(regression_tree)

        f_k_matrix = update_f_k_matrix(regression_tree, f_k_minus_1_matrix)
        f_k_minus_1_matrix = f_k_matrix

    return f_k_minus_1_matrix, model_regression_trees, f_0_matrix

def calculate_residuals_matrix(S_delta_matrix, f_k_minus_1_matrix):
    return S_delta_matrix - f_k_minus_1_matrix

def calculate_f_0_matrix(S_delta_matrix):
    return np.mean(S_delta_matrix, axis=0) #TODO Correct to use mean?

def update_f_k_matrix(regression_tree, f_k_minus_1_matrix, learning_rate=_LEARNING_RATE):
    g_k_matrix = regression_tree.get_avarage_residuals_matrix()
    f_k_matrix = f_k_minus_1_matrix + learning_rate * g_k_matrix
    return f_k_matrix
