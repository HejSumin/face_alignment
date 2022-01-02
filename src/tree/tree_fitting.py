import numpy as np
from src.tree.regression_tree import *
from timeit import default_timer as timer
from datetime import timedelta
from tqdm import tqdm

_DEBUG = True
_DEBUG_DETAILED = False
_DEBUG_GRAPHVIZ = True

"""
Hyperparameters

Parameters
    ----------
    _AMOUNT_RANDOM_CANDIDATE_SPLITS : #amount of random candidate splits generated for each node
    _REGRESSION_TREE_MAX_DEPTH : depth of the regression tree -> e.g. depth 5 results in 32 leaf nodes
"""
_AMOUNT_RANDOM_CANDIDATE_SPLITS = 20
_EXPONENTIAL_PRIOR_LAMBDA = -0.05

"""
Select and return best candidate split (pixel x1, pixel x2, pixel intensity threshold) for a single node.

Parameters
    ----------
    I_grayscale_image_matrix : matrix with rows of grayscale image values for extraced features/pixels
        Note: one row for each I (image) in each (I, S_hat, S_delta) triplet of our data

    residual_image_vector_matrix : matrix with rows of residual image vectors (all (194) resdiuals computed for a single image and data triplet (I, S_hat, S_delta))
        Note: amount and order of images / triplets (I, S_hat, S_delta) must be the same as in I_grayscale_image_matrix

    theta_candidate_splits : possible theta (pixel x1, pixel x2, pixel intensity threshold) candidate splits

    Q_I_at_node : set of indicies of images (I) for each triplet that are bucketized at current node

    mu_parent_node : mu value (avarage residual values) at parent node
        Note: parameter is None when selecting best candidate split for root node

Returns
    -------
    (pixel x1, pixel x2, pixel intensity threshold), mu_theta : best candidate split triplet and corresponding Q_theta_l, Q_thetas_r, mu_theta value needed for calculation in the next iteration
"""
def _select_best_candidate_split_for_node(I_intensities_matrix, residuals_matrix, features_hat_matrix, Q_I_at_node, mu_parent_node=None, use_exponential_prior=True):
    features_hat_at_Node_matrix = features_hat_matrix[Q_I_at_node]
    features_hat_mean_coords = np.mean(features_hat_at_Node_matrix, axis=0) if features_hat_at_Node_matrix.shape[0] > 0 else None
    theta_candidate_splits = _generate_random_candidate_splits(I_intensities_matrix.shape[1], features_hat_mean_coords=features_hat_mean_coords, use_exponential_prior=use_exponential_prior)

    sum_square_error_theta_candidate_splits = np.zeros((theta_candidate_splits.shape[0], 1))
    mu_thetas = []
    Q_thetas_l = []
    Q_thetas_r = []

    for i, theta in enumerate(theta_candidate_splits):
        x1, x2, threshold = theta[0], theta[1], theta[2]
        Q_theta_l = []
        Q_theta_r = []

        # bucketize images based on theta candidate split
        for index in Q_I_at_node:
            if np.abs(I_intensities_matrix[index, x1].astype(np.int16) - I_intensities_matrix[index, x2].astype(np.int16)) > threshold:
                Q_theta_l.append(index)
            else:
                Q_theta_r.append(index)

        mu_theta_l = (len(Q_theta_l) and 1 / len(Q_theta_l) or 0) * np.sum(residuals_matrix[Q_theta_l], axis=0, dtype=np.float32)
        mu_theta_r = None # np.empty(residuals_matrix[Q_theta_r].shape)
        if mu_parent_node is None: # True if selecting candidate split for root node
            mu_theta_r = (len(Q_theta_r) and 1 / len(Q_theta_r) or 0) * np.sum(residuals_matrix[Q_theta_r], axis=0, dtype=np.float32)
        else:
            mu_theta_r = (len(Q_theta_r) and 1 / len(Q_theta_r) or 0) * (len(Q_I_at_node) * mu_parent_node -  len(Q_theta_l) * mu_theta_l)

        sum_square_error_theta = (len(Q_theta_l) * np.matmul(mu_theta_l.T, mu_theta_l)) + (len(Q_theta_r) * np.matmul(mu_theta_r.T, mu_theta_r))
        sum_square_error_theta_candidate_splits[i] = sum_square_error_theta
        mu_thetas.append((mu_theta_l.astype(np.float32) , mu_theta_r.astype(np.float32)))
        Q_thetas_l.append(Q_theta_l)
        Q_thetas_r.append(Q_theta_r)

    best_theta_candidate_split_index = np.argmax(sum_square_error_theta_candidate_splits)
    return theta_candidate_splits[best_theta_candidate_split_index],  Q_thetas_l[best_theta_candidate_split_index], Q_thetas_r[best_theta_candidate_split_index], mu_thetas[best_theta_candidate_split_index]

def _generate_random_candidate_splits(amount_extraced_features, features_hat_mean_coords=None, amount_candidate_splits=_AMOUNT_RANDOM_CANDIDATE_SPLITS, use_exponential_prior=True):
    random_candidate_splits = np.empty((amount_candidate_splits, 3), dtype=int)
    for i in range(0, amount_candidate_splits):
        while True:
            random_x1_pixel_index = np.random.randint(0, amount_extraced_features)
            random_x2_pixel_index = np.random.randint(0, amount_extraced_features)

            if use_exponential_prior and features_hat_mean_coords is not None:
                u = np.array([features_hat_mean_coords[random_x1_pixel_index*2], features_hat_mean_coords[random_x1_pixel_index*2+1]])
                v = np.array([features_hat_mean_coords[random_x2_pixel_index*2], features_hat_mean_coords[random_x2_pixel_index*2+1]])
                pixel_distance = np.absolute(np.linalg.norm(u-v))
                probability = np.exp(pixel_distance*_EXPONENTIAL_PRIOR_LAMBDA)
                if random_x1_pixel_index != random_x2_pixel_index and probability > np.random.random():
                    break
            else:
                if random_x1_pixel_index != random_x2_pixel_index:
                    break

        random_threshold = np.random.randint(0, 256) # we take the absolute value for the pixel intensity difference (0-255)
        random_candidate_splits[i] = np.array([random_x1_pixel_index, random_x2_pixel_index, random_threshold])
    return random_candidate_splits

def _generate_root_node(regression_tree, I_intensities_matrix, residuals_matrix, features_hat_matrix, Q_I_at_root, use_exponential_prior):
    (best_x1_pixel_index_root, best_x2_pixel_index_root, best_threshold_root), Q_theta_l_root, Q_theta_r_root, mu_theta_root = _select_best_candidate_split_for_node(
        I_intensities_matrix,
        residuals_matrix,
        features_hat_matrix,
        Q_I_at_root,
        use_exponential_prior
    )
    return regression_tree.create_node(best_x1_pixel_index_root, best_x2_pixel_index_root, best_threshold_root), Q_theta_l_root, Q_theta_r_root, mu_theta_root

def _generate_leaf_node(regression_tree, avarage_residual_vector, parent_id):
    return regression_tree.create_leaf(avarage_residual_vector, parent_id)

def _generate_child_nodes(
        regression_tree,
        current_node_id,
        current_depth,
        max_depth,
        I_intensities_matrix,
        residuals_matrix,
        features_hat_matrix,
        Q_theta_l,
        Q_theta_r,
        mu_parent_node,
        use_exponential_prior
    ):
    mu_theta_l, mu_theta_r = mu_parent_node

    if current_depth == max_depth-1:
        _generate_leaf_node(regression_tree, mu_theta_l, parent_id=current_node_id)
        _generate_leaf_node(regression_tree, mu_theta_r, parent_id=current_node_id)
        regression_tree.append_avarage_residuals_matrix(mu_theta_l, Q_theta_l) # used for training as result of g_k
        regression_tree.append_avarage_residuals_matrix(mu_theta_r, Q_theta_r) # used for training as result of g_k
        return True

    (best_x1_pixel_index_left_child, best_x2_pixel_index_left_child, best_threshold_left_child), Q_theta_l_left_child, Q_theta_r_left_child, mu_theta_left_child = _select_best_candidate_split_for_node(
        I_intensities_matrix,
        residuals_matrix,
        features_hat_matrix,
        Q_theta_l,
        mu_theta_l,
        use_exponential_prior
    )

    (best_x1_pixel_index_right_child, best_x2_pixel_index_right_child, best_threshold_right_child), Q_theta_l_right_child, Q_theta_r_right_child, mu_theta_right_child = _select_best_candidate_split_for_node(
        I_intensities_matrix,
        residuals_matrix,
        features_hat_matrix,
        Q_theta_r,
        mu_theta_r,
        use_exponential_prior
    )

    # we are always creating two new nodes at a time
    left_node = regression_tree.create_node(best_x1_pixel_index_left_child, best_x2_pixel_index_left_child, best_threshold_left_child, parent_id=current_node_id)  # left node, has parent current_node
    right_node = regression_tree.create_node(best_x1_pixel_index_right_child, best_x2_pixel_index_right_child, best_threshold_right_child, parent_id=current_node_id)  # right node, has parent current_node

    return (
        _generate_child_nodes(
            regression_tree,
            left_node.id,
            current_depth+1,
            max_depth,
            I_intensities_matrix,
            residuals_matrix,
            features_hat_matrix,
            Q_theta_l_left_child,
            Q_theta_r_left_child,
            mu_theta_left_child,
            use_exponential_prior
        ), _generate_child_nodes(
            regression_tree,
            right_node.id,
            current_depth+1,
            max_depth,
            I_intensities_matrix,
            residuals_matrix,
            features_hat_matrix,
            Q_theta_l_right_child,
            Q_theta_r_right_child,
            mu_theta_right_child,
            use_exponential_prior
        )
    )

def generate_regression_tree(I_intensities_matrix, residuals_matrix, features_hat_matrix, regression_tree_max_depth=5, use_exponential_prior=True):
    Q_I_at_root = np.arange(0, I_intensities_matrix.shape[0])

    regression_tree = Regression_Tree(avarage_residuals_matrix_shape=residuals_matrix.shape)
    root_node, Q_theta_l_root, Q_theta_r_root, mu_theta_root = _generate_root_node(regression_tree, I_intensities_matrix, residuals_matrix, features_hat_matrix, Q_I_at_root, use_exponential_prior)

    success = _generate_child_nodes(regression_tree, root_node.id, 0, regression_tree_max_depth, I_intensities_matrix, residuals_matrix, features_hat_matrix, Q_theta_l_root, Q_theta_r_root, mu_theta_root, use_exponential_prior)
    return regression_tree

_NUMBER_SPLIT_VALUES_AT_NODE = 3
_AMOUNT_LANDMARKS_FLATTENED = 388
def predict_avarage_residual_vector_for_image(regression_tree_vector, avarage_residual_leaf_vector, I_intensities):
    max_depth = get_max_depth_by_node_number(regression_tree_vector.shape[0] / _NUMBER_SPLIT_VALUES_AT_NODE)

    current_node_index = 0
    current_depth = 0
    depth_to_leafs = max_depth - current_depth 

    while depth_to_leafs > 0:
        x1 = regression_tree_vector[current_node_index]
        x2 = regression_tree_vector[current_node_index+1]
        threshold = regression_tree_vector[current_node_index+2]

        if current_node_index == 0:
            left_node_index = 1 * _NUMBER_SPLIT_VALUES_AT_NODE
        else:
            left_node_index = (current_node_index + (2**(depth_to_leafs+1) - 1) * _NUMBER_SPLIT_VALUES_AT_NODE) if current_node_index % 2 == 0 else current_node_index + 2 * _NUMBER_SPLIT_VALUES_AT_NODE
        right_node_index = left_node_index + 1 * _NUMBER_SPLIT_VALUES_AT_NODE

        if np.abs(I_intensities[x1].astype(np.int16) - I_intensities[x2].astype(np.int16)) > threshold: # True == go to left node; False == go to right node
            current_node_index = left_node_index
        else:
            current_node_index = right_node_index
        
        current_depth = current_depth + 1
        depth_to_leafs = max_depth - current_depth 

    leaf_index = regression_tree_vector[current_node_index] * _AMOUNT_LANDMARKS_FLATTENED
    return avarage_residual_leaf_vector[leaf_index:(leaf_index  +_AMOUNT_LANDMARKS_FLATTENED)]

def get_max_depth_by_node_number(amount_nodes):
    return int(np.floor(np.log2(amount_nodes)))

def save_regression_trees_to_file(model_regression_trees, output_path, t, regression_tree_max_depth):
    amount_regression_trees = len(model_regression_trees)
    amount_leafs_per_regression_tree = 2**regression_tree_max_depth
    amount_nodes_per_regression_tree = len(model_regression_trees[0].get_nodes_list()) - amount_leafs_per_regression_tree
    amount_landmarks_flattened = model_regression_trees[0].get_nodes_list()[-1].avarage_residual_vector.shape[0]

    model_avarage_residual_leaf_matrix = np.empty((amount_regression_trees, amount_leafs_per_regression_tree*amount_landmarks_flattened), dtype=np.float32)
    model_regression_trees_matrix = np.empty((amount_regression_trees, (amount_nodes_per_regression_tree+amount_leafs_per_regression_tree)*_NUMBER_SPLIT_VALUES_AT_NODE), dtype=np.uint16)

    for i, regression_tree in tqdm(enumerate(model_regression_trees), desc="Saving tree model"):
        regression_tree_vector = build_regression_tree_vector(regression_tree.get_nodes_list())
        model_regression_trees_matrix[i] = regression_tree_vector.flatten()

        regression_tree_leafs = filter(lambda x: isinstance(x, Leaf), regression_tree.get_nodes_list())
        regression_leafs_vector = np.array([leaf.avarage_residual_vector for leaf in regression_tree_leafs], dtype=np.float32)
        model_avarage_residual_leaf_matrix[i] = regression_leafs_vector.flatten()

    np.save(output_path + "model_regression_trees_matrix_cascade_" + str(t), model_regression_trees_matrix)
    np.save(output_path + "model_avarage_residual_leaf_matrix_cascade_" + str(t), model_avarage_residual_leaf_matrix)    

def build_regression_tree_vector(regression_tree_node_list):
    regression_tree_vector = np.empty((1, len(regression_tree_node_list) * _NUMBER_SPLIT_VALUES_AT_NODE), dtype=np.uint16)
    leaf_number = 0
    for k, node in enumerate(regression_tree_node_list):
        index = k*_NUMBER_SPLIT_VALUES_AT_NODE
        if isinstance(node, Leaf):
            regression_tree_vector[:,index:index+_NUMBER_SPLIT_VALUES_AT_NODE] = [leaf_number, 0, 0]
            leaf_number = leaf_number + 1
        else:
            regression_tree_vector[:,index:index+_NUMBER_SPLIT_VALUES_AT_NODE] = [node.x1, node.x2, node.threshold]
    return regression_tree_vector

def predict_avarage_residual_vector_for_image_from_regression_tree_object(regression_tree, I_intensities, current_node_id=None):
    current_node = None
    if current_node_id is None:
        current_node = regression_tree.get_root_node()
    else:
        current_node = regression_tree.find_node_by_id(current_node_id)

    if  isinstance(current_node, Leaf):
        return current_node.avarage_residual_vector
    else:
        if np.abs(I_intensities[current_node.x1].astype(np.int16) - I_intensities[current_node.x2].astype(np.int16)) > current_node.threshold:
            return predict_avarage_residual_vector_for_image_from_regression_tree_object(regression_tree, I_intensities, current_node.left_child_id)
        else:
            return predict_avarage_residual_vector_for_image_from_regression_tree_object(regression_tree, I_intensities, current_node.right_child_id)

def run_test_example():
    images = 2000
    landmarks = 194
    R = 20
    n_image_matrix = np.random.randint(0, 256, (images, 20))
    I_intensities_matrix = np.repeat(n_image_matrix, repeats=R, axis=0) # shape (N=n*R, #extraced pixels)
    residuals_matrix = np.random.rand(R*images, landmarks*2) # only positive values for test example ; shape (N=n*R, 194)
    features_hat_matrix = np.random.rand(R*images, landmarks*2)

    start = timer()
    regression_tree = generate_regression_tree(I_intensities_matrix, residuals_matrix, features_hat_matrix)
    end = timer()

    if _DEBUG:
        if _DEBUG_DETAILED:
            print("I_intensities_matrix : " + str(I_intensities_matrix))
            print("residuals_matrix : " + str(residuals_matrix))
        print()
        print(regression_tree.get_tree_description(detailed=_DEBUG_DETAILED))
        print([node.id for node in regression_tree.get_nodes_list()])
        print("Time: ", timedelta(seconds=end-start))
        if _DEBUG_GRAPHVIZ:
            graphviz = regression_tree.get_dot_graphviz_source()
            graphviz_file = open('./graphviz_output.txt', 'w', encoding='utf-8')
            graphviz_file.write(graphviz)
            graphviz_file.close()
