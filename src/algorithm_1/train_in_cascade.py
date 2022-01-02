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

def train_multiple_cascades(training_data, regression_tree_max_depth=5, use_exponential_prior=True):
    I_intensities_matrix, features_hat_matrix, S_hat_matrix, S_delta_matrix, S_true_matrix = prepare_training_data_for_tree_cascade(training_data)

    for t in tqdm(range(0, _T), desc="T cascades"):
        last_run = t == _T-1

        r_t_matrix, model_regression_trees, f_0_matrix = train_single_cascade(I_intensities_matrix, features_hat_matrix, S_delta_matrix, regression_tree_max_depth, use_exponential_prior)
        np.save("run_output/run_output_model_f_0_matrix" + str(t), f_0_matrix, allow_pickle=True)

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

def save_regression_trees(model_regression_trees, t, regression_tree_max_depth):
    amount_regression_trees = len(model_regression_trees)
    amount_leafs_per_regression_tree = 2**regression_tree_max_depth
    amount_nodes_per_regression_tree = len(model_regression_trees[0].get_nodes_list()) - amount_leafs_per_regression_tree
    amount_landmarks_flattened = model_regression_trees[0].get_nodes_list()[-1].avarage_residual_vector.shape[0]


    model_avarage_residual_leaf_matrix = np.empty((amount_regression_trees, amount_leafs_per_regression_tree*amount_landmarks_flattened))
    model_regression_trees_matrix = np.empty((amount_regression_trees, amount_nodes_per_regression_tree*3))

    for i, regression_tree in enumerate(model_regression_trees):
        regression_tree_without_leafs = filter(lambda x: not isinstance(x, Leaf), regression_tree.get_nodes_list())
        regression_tree_vector = np.array([[node.x1, node.x2, node.threshold] for node in regression_tree_without_leafs], dtype=np.uint16)
        model_regression_trees_matrix[i] = regression_tree_vector.flatten()
        regression_tree_leafs = filter(lambda x: isinstance(x, Leaf), regression_tree.get_nodes_list())
        regression_leafs_vector = np.array([leaf.avarage_residual_vector for leaf in regression_tree_leafs], dtype=np.uint16)
        model_avarage_residual_leaf_matrix[i] = regression_leafs_vector.flatten()

    np.save("run_output/run_output_model_regression_trees_matrix_cascade_" + str(t), model_regression_trees_matrix)
    np.save("run_output/run_output_avarage_residual_leaf_matrix_cascade_" + str(t), model_avarage_residual_leaf_matrix, allow_pickle=True)
