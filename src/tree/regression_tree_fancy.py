import numpy as np

class Regression_Tree_Fancy:

    def __init__(self, avarage_residuals_matrix_shape):
        self.regression_tree_vector = []
        self.regression_leafs_vector = []
        self.current_leaf_count = 0
        self._avarage_residuals_matrix = np.empty(avarage_residuals_matrix_shape)

    def create_node(self, x1, x2, threshold):
        self.regression_tree_vector.append(x1)
        self.regression_tree_vector.append(x2)
        self.regression_tree_vector.append(threshold)

    def create_leaf(self, avarage_residual_vector):
        self.regression_tree_vector.append(self.current_leaf_count)
        self.regression_tree_vector.append(0)
        self.regression_tree_vector.append(0)        
        self.regression_leafs_vector.append(avarage_residual_vector)
        self.current_leaf_count = self.current_leaf_count + 1
    
    def get_regression_tree_vector(self):
        return self.regression_tree_vector

    def get_regression_leafs_vector(self):
        return self.regression_leafs_vector

    def get_avarage_residuals_matrix(self):
        return self._avarage_residuals_matrix

    def append_avarage_residuals_matrix(self, avarage_residual_vector, Q_I_at_node):
        self._avarage_residuals_matrix[Q_I_at_node] = avarage_residual_vector
