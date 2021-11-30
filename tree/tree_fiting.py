import numpy as np
import uuid

_DEBUG = True
_AMOUNT_CADIDATE_SPLITS = 20
_AMOUND_EXTRACTED_FEATURES = 400
_REGRESSION_TREE_MAX_DEPTH = 5

_INSERT, _DELETE = range(2)

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

    def get_node_description(self):
        return '[id: %s, x1: %d, x2: %d, threshold: %d, left child id: %s, right child id: %s]' % (self.id, self.x1, self.x2, self.threshold, self.left_child_id, self.right_child_id)

class Regression_Tree:

    def __init__(self):
        self.nodes = []

    def create_node(self, x1, x2, threshold, parent_id=None):
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

    def get_tree_description(self):
        result = "<< 🌳 regression tree 🌳 >>\n"
        for node in self.nodes:
            result += node.get_node_description() + "\n"
        return result + "\n<< 🌳 regression tree 🌳 >>\n"

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
    (pixel x1, pixel x2, pixel intensity threshold), mu_theta : best candidate split triplet and corresponding Q_theta_l, Q_thetas_r, mu_theta value needed for calculation in the next iteration
"""
def select_best_candidate_split_for_node(I_grayscale_image_matrix, residual_image_vector_matrix, theta_candidate_splits, Q_images_at_node, mu_parent_node=None):
    sum_square_error_theta_candidate_splits = []
    mu_thetas = []
    Q_thetas_l =  []
    Q_thetas_r = []

    for theta in theta_candidate_splits:
        x1, x2, threshold = theta
        Q_theta_l = []
        Q_theta_r = []

        # bucketize images based on theta candidate split
        for index in Q_images_at_node:
            if np.abs(I_grayscale_image_matrix[index][x1] - I_grayscale_image_matrix[index][x2]) > threshold: 
                Q_theta_l.append(index)
            else:
                Q_theta_r.append(index)

        mu_theta_l = (len(Q_theta_l) and 1 / len(Q_theta_l) or 0) * np.sum(residual_image_vector_matrix[Q_theta_l], axis=0) 
        mu_theta_r = np.empty(residual_image_vector_matrix[Q_theta_r].shape)
        if mu_parent_node is None: # True if selecting candidate split for root node
            mu_theta_r = (len(Q_theta_r) and 1 / len(Q_theta_r) or 0) * np.sum(residual_image_vector_matrix[Q_theta_r], axis=0)
        else:
            mu_theta_r = (len(Q_theta_r) and 1 / len(Q_theta_r) or 0) * (len(Q_images_at_node) * mu_parent_node - len(Q_theta_l) * mu_theta_l)
    
        sum_square_error_theta = (len(Q_theta_l) * np.matmul(mu_theta_l.T, mu_theta_l)) + (len(Q_theta_r) * np.matmul(mu_theta_r.T, mu_theta_r))
        sum_square_error_theta_candidate_splits.append(sum_square_error_theta)
        mu_thetas.append(mu_theta_l + mu_theta_r)
        Q_thetas_l.append(Q_theta_l)
        Q_thetas_r.append(Q_theta_r)

    best_theta_candidate_split_index = np.argmax(sum_square_error_theta_candidate_splits)
    return theta_candidate_splits[best_theta_candidate_split_index],  Q_thetas_l[best_theta_candidate_split_index], Q_thetas_l[best_theta_candidate_split_index], mu_thetas[best_theta_candidate_split_index]

def generate_random_candidate_splits(amount_candidate_splits=_AMOUNT_CADIDATE_SPLITS, amount_extraced_features=_AMOUND_EXTRACTED_FEATURES):
    random_candidate_splits = []
    for _ in range(0, amount_candidate_splits): 
        random_x1_pixel_index = np.random.randint(0, amount_extraced_features)
        random_x2_pixel_index = np.random.randint(0, amount_extraced_features)
        while (random_x1_pixel_index == random_x2_pixel_index):
            random_x2_pixel_index = np.random.randint(0, amount_extraced_features)

        random_threshold = np.random.randint(0, 256) # we take the absolute value for the pixel intensity differnece (0-255)
        random_candidate_splits.append((random_x1_pixel_index, random_x2_pixel_index, random_threshold))
    return random_candidate_splits

def generate_root_node(I_grayscale_image_matrix, residual_image_vector_matrix, Q_images_at_node):
    random_candidate_splits_root = generate_random_candidate_splits()
    (best_x1_pixel_index_root, best_x2_pixel_index_root, best_threshold_root), Q_theta_l_root, Q_theta_r_root, mu_theta_root = select_best_candidate_split_for_node(
        I_grayscale_image_matrix,
        residual_image_vector_matrix,
        random_candidate_splits_root,
        Q_images_at_node
    )
    return _REGRESSION_TREE.create_node(best_x1_pixel_index_root, best_x2_pixel_index_root, best_threshold_root), Q_theta_l_root, Q_theta_r_root, mu_theta_root 

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
        return True

    random_candidate_splits_left_child = generate_random_candidate_splits()
    (best_x1_pixel_index_left_child, best_x2_pixel_index_left_child, best_threshold_left_child), Q_theta_l_left_child, Q_theta_r_left_child, mu_theta_left_child = select_best_candidate_split_for_node(
        I_grayscale_image_matrix,
        residual_image_vector_matrix,
        random_candidate_splits_left_child,
        Q_theta_l,
        mu_parent_node
    )

    random_candidate_splits_right_child = generate_random_candidate_splits()
    (best_x1_pixel_index_right_child, best_x2_pixel_index_right_child, best_threshold_right_child), Q_theta_l_right_child, Q_theta_r_right_child, mu_theta_right_child = select_best_candidate_split_for_node(
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

# TODO restirct depth by setting minimum amount of images bucketized in one node/leaf
# TODO remove candidate split calculation for leaf nodes
# TODO calculate and save avarage residual values (delta landmarks) for leaf nodes
# TODO build function to search trough the regression tree in order to find correct landmark delta values for each Image

I_grayscale_image_matrix = np.random.randint(0, 256, (20, 400)) # shape (n, #extraced pixels)
residual_image_vector_matrix = np.random.rand(20, 20) # only positive values for test example ; shape (n, R) #TODO should be actualy of shape (n, R *(1, 194))
Q_images_at_node = [i for i in range(0, 20)]

if _DEBUG:
    print("I_grayscale_image_matrix : " + str(I_grayscale_image_matrix))
    print("residual_image_vector_matrix : " + str(residual_image_vector_matrix))
    print("Q_images_at_node : " + str(Q_images_at_node))

_REGRESSION_TREE = Regression_Tree()
root_node, Q_theta_l_root, Q_theta_r_root, mu_theta_root = generate_root_node(I_grayscale_image_matrix, residual_image_vector_matrix, Q_images_at_node)

if _DEBUG:
    print()
    print("root node: " + root_node.get_node_description())

regression_tree_generation_successful = generate_child_nodes(root_node.id, 0, _REGRESSION_TREE_MAX_DEPTH, I_grayscale_image_matrix, residual_image_vector_matrix, Q_theta_l_root, Q_theta_r_root, mu_theta_root)
print("🌳 regression tree successfully generated ... ", regression_tree_generation_successful)

if _DEBUG:
    print()
    print(_REGRESSION_TREE.get_tree_description())