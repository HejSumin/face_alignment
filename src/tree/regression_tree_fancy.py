import numpy as np

_AMOUNT_LANDMARKS_FLATTENED = 388
_NUMBER_SPLIT_VALUES_AT_NODE = 3

class Regression_Tree_Fancy:

    def __init__(self, avarage_residuals_matrix_shape, regression_tree_max_depth):
        self.current_node_index = 0
        self.current_leaf_index = 0
        self.regression_tree_vector = np.empty((1, (2**(regression_tree_max_depth+1) - 1) * _NUMBER_SPLIT_VALUES_AT_NODE), dtype=np.uint16)
        self.regression_leafs_vector = np.empty((2**regression_tree_max_depth, _AMOUNT_LANDMARKS_FLATTENED), dtype=np.float32)
        self._avarage_residuals_matrix = np.empty(avarage_residuals_matrix_shape)

    def create_node(self, x1, x2, threshold):
        self.regression_tree_vector[:,self.current_node_index] = x1
        self.regression_tree_vector[:,self.current_node_index+1] = x2
        self.regression_tree_vector[:,self.current_node_index+2] = threshold
        self.current_node_index = self.current_node_index + _NUMBER_SPLIT_VALUES_AT_NODE

    def create_leaf(self, avarage_residual_vector):
        self.regression_tree_vector[:,self.current_node_index] = self.current_leaf_index
        self.regression_tree_vector[:,self.current_node_index+1] = 0
        self.regression_tree_vector[:,self.current_node_index+2] = 0   
        self.regression_leafs_vector[self.current_leaf_index] = avarage_residual_vector
        self.current_node_index = self.current_node_index + _NUMBER_SPLIT_VALUES_AT_NODE
        self.current_leaf_index = self.current_leaf_index + 1
    
    def get_regression_tree_vector(self):
        return self.regression_tree_vector

    def get_regression_leafs_vector(self):
        return self.regression_leafs_vector.flatten()

    def get_avarage_residuals_matrix(self):
        return self._avarage_residuals_matrix

    def append_avarage_residuals_matrix(self, avarage_residual_vector, Q_I_at_node):
        self._avarage_residuals_matrix[Q_I_at_node] = avarage_residual_vector
