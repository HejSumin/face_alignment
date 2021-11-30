import numpy as np
import uuid

from numpy.core.fromnumeric import argmax
number_selected_features = 400

(_INSERT, _DELETE) = range(2)

class Node:

    def __init__(self, x1, x2, threshold):
        self.id = uuid.uuid4()
        self.x1 = x1
        self.x2 = x2
        self.threshold = threshold
        self.left_child_id = None
        self.right_child_id = None

    def update_child(self, child_id, left=True, mode=_INSERT):
        if mode is _INSERT:
            if left:
                self.left_child_id = child_id 
            else:
                self.right_child_id = child_id 
        elif mode is _DELETE:
            if left:
                self.left_child_id = None
            else:
                self.right_child_id = None

    def print_node(self):
        return '[id %s, x1 %d, x2 %d, threshold %d, left %s, right %s]' % (self.id, self.x1, self.x2, self.threshold, self.left_child_id, self.right_child_id)

class Regression_Tree:

    def __init__(self):
        self.nodes = []

    def create_node(self, x1, x2, threshold, parent_id=None):
        print("parent " + str(parent_id))
        node = Node(x1, x2, threshold)
        self.nodes.append(node)
        self.__update_childs(parent_id, node.id, _INSERT)
        return node

    def __update_childs(self, position, id, mode):
        if position is None:
            return
        else:
            if self[position].left_child_id == None:
                self[position].update_child(child_id=id, left=True, mode=mode)
            else:
                self[position].update_child(child_id=id, left=False, mode=mode)

    def print_tree(self):
        for node in self.nodes:
            print(node.print_node())

    def get_index(self, position):
        for index, node in enumerate(self.nodes):
            if node.id == position:
                break
        return index

    def __getitem__(self, key):
        return self.nodes[self.get_index(key)]

    def __setitem__(self, key, item):
        self.nodes[self.get_index(key)] = item

"""
Select and return best candidate split (pixel x1, pixel x2, pixel intensity threshold) for a single node.

Parameters
    ----------
    I_grayscale_image_matrix :  matrix with rows of grayscale image values for extraced features/pixels

    residual_image_vector_matrix : matrix with rows of residual image vectors (all resdiuals computed for a single image) 
        Note: shape and order of images must be the same as in I_grayscale_image_matrix

    theta_candidate_splits : possible theta (pixel x1, pixel x2, pixel intensity threshold) candidate splits

    Q_images_at_node : set of indicies of images that are bucketized at current node

    mu_parent_node : mu value (avarage residual values) at parent node
        Note: parameter is None when selecting best candidate split for root node

Returns
    -------
    (pixel x1, pixel x2, pixel intensity threshold), mu_theta : best candidate split triplet and corresponding mu_theta value to simplify calculation for next iteration
"""
def select_best_candidate_split_for_node(I_grayscale_image_matrix, residual_image_vector_matrix, theta_candidate_splits, Q_images_at_node, mu_parent_node=None):
    sum_square_error_theta_candidate_splits = np.empty(len(theta_candidate_splits))
    mu_thetas = np.empty(len(theta_candidate_splits))
    Q_thetas_l = np.empty(len(theta_candidate_splits))
    Q_thetas_r = np.empty(len(theta_candidate_splits))

    for theta in theta_candidate_splits:
        x1, x2, threshold = theta
        Q_theta_l, Q_theta_r = []

        # bucketize images based on theta candidate split
        for index in Q_images_at_node:
            if np.abs(I_grayscale_image_matrix[index][x1] - I_grayscale_image_matrix[index][x2]) > threshold: 
                Q_theta_l.append(index)
            else:
                Q_theta_r.append(index)

        mu_theta_l = 1 / len(Q_theta_l) * np.sum(residual_image_vector_matrix[Q_theta_l])
        if mu_parent_node == None: # True if selecting candidate split for root node
            mu_theta_r = 1 / len(Q_theta_r) * np.sum(residual_image_vector_matrix[Q_theta_r])
        else:
            mu_theta_r = 1/ len(Q_theta_r) * (len(Q_images_at_node) * mu_parent_node - len(Q_theta_l) * mu_theta_l)
            
        sum_square_error_theta = (len(Q_theta_l) * np.matmul(mu_theta_l.T, mu_theta_l)) + (len(Q_theta_r) * np.matmul(mu_theta_r.T, mu_theta_r))
        sum_square_error_theta_candidate_splits.append(sum_square_error_theta)
        mu_thetas.append(mu_theta_l + mu_theta_r)
        Q_thetas_l.append(Q_theta_l)
        Q_thetas_r.append(Q_theta_r)

    best_theta_candidate_split_index = argmax(sum_square_error_theta_candidate_splits)
    return theta_candidate_splits[best_theta_candidate_split_index],  Q_thetas_l[best_theta_candidate_split_index], Q_thetas_l[best_theta_candidate_split_index], mu_thetas[best_theta_candidate_split_index]

def generate_random_candidate_splits(amount_candidate_splits=20):
    random_candidate_splits = []
    for _ in range(0, amount_candidate_splits-1): 
        random_x1_pixel_index = np.random.randint(0, number_selected_features)
        random_x2_pixel_index = np.random.randint(0, number_selected_features)
        while (random_x1_pixel_index == random_x2_pixel_index):
            random_x2_pixel_index = np.random.randint(0, number_selected_features)

        random_threshold = np.random.randint(0, 255) # we take the absolute value for the pixel intensity differnece

        random_candidate_splits.append(zip(random_x1_pixel_index, random_x2_pixel_index, random_threshold))
    return random_candidate_splits

def generate_root_node(I_grayscale_image_matrix, residual_image_vector_matrix, Q_images_at_node):
    random_candidate_splits_root = generate_random_candidate_splits()
    best_x1_pixel_index_root, best_x2_pixel_index_root, best_threshold_root, Q_theta_l, Q_theta_r, mu_theta_root = select_best_candidate_split_for_node(
        I_grayscale_image_matrix,
        residual_image_vector_matrix,
        random_candidate_splits_root,
        Q_images_at_node
    )
    return _REGRESSION_TREE.create_node(best_x1_pixel_index_root, best_x2_pixel_index_root, best_threshold_root), Q_theta_l, Q_theta_r, mu_theta_root 

def generate_child_nodes(
        current_node_id, 
        current_depth, 
        max_depth, 
        I_grayscale_image_matrix, 
        residual_image_vector_matrix, 
        Q_theta_l,
        Q_theta_r,
        mu_parent_node
    ):
    if current_depth == max_depth:
        return

    random_candidate_splits_left_child = generate_random_candidate_splits()
    best_x1_pixel_index_left_child, best_x2_pixel_index_left_child, best_threshold_left_child, Q_theta_l_left_child, Q_theta_r_left_child, mu_theta_left_child = select_best_candidate_split_for_node(
        I_grayscale_image_matrix,
        residual_image_vector_matrix,
        random_candidate_splits_left_child,
        Q_theta_l,
        mu_parent_node
    )

    random_candidate_splits_right_child = generate_random_candidate_splits()
    best_x1_pixel_index_right_child, best_x2_pixel_index_right_child, best_threshold_right_child, Q_theta_l_right_child, Q_theta_r_right_child, mu_theta_right_child = select_best_candidate_split_for_node(
        I_grayscale_image_matrix,
        residual_image_vector_matrix,
        random_candidate_splits_right_child,
        Q_theta_r,
        mu_parent_node
    )

    # we are always creating two new nodes at a time
    left_node = _REGRESSION_TREE.create_node(best_x1_pixel_index_left_child, best_x2_pixel_index_left_child, best_threshold_left_child, parent_id=current_node_id)  # left node, has parent current_node
    right_node = _REGRESSION_TREE.create_node(best_x1_pixel_index_right_child, best_x2_pixel_index_right_child, best_threshold_right_child, parent_id=current_node_id)  # right node, has parent current_node

    return (
        generate_child_nodes(
            left_node.id, 
            current_depth+1, 
            max_depth, 
            I_grayscale_image_matrix,
            residual_image_vector_matrix,
            Q_theta_l_left_child,
            Q_theta_r_left_child,
            mu_theta_left_child
        ), generate_child_nodes(
            right_node.id, 
            current_depth+1, 
            max_depth, 
            I_grayscale_image_matrix,
            residual_image_vector_matrix,
            Q_theta_l_right_child,
            Q_theta_r_right_child,
            mu_theta_right_child
        )
    ) 

# try running 
# restirct depth by setting minimum amount of images buckitized in one node!

_REGRESSION_TREE = Regression_Tree()
root = generate_root(tree)
generate_complete_node(current_node_id=root.id, current_depth=0, max_depth=2)

# print(len(tree.nodes))
# print(tree.print_tree())