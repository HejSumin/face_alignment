from src.face_alignment.utility import *
from src.tree.tree_fitting import *

class SingleCascade():

    def __init__(self, model_regression_trees_matrix, model_avarage_resiudal_leaf_matrix, model_f_0_matrix, model_learning_rate):
        self.model_regression_trees_matrix = model_regression_trees_matrix
        self.model_avarage_resiudal_leaf_matrix = model_avarage_resiudal_leaf_matrix
        self.model_f_0_matrix = model_f_0_matrix
        self.model_learning_rate = model_learning_rate

    def apply_cascade(self, I, S_hat, features_hat, S_mean, features_mean):
        cascade_contribution = self.model_f_0_matrix
        x_mask = [x for x in range(0, cascade_contribution.shape[0]-1, 2)]
        y_mask = [y for y in range(1, cascade_contribution.shape[0], 2)]
    
        I_intensities = I[np.array(features_hat[:,1]), np.array(features_hat[:,0])].astype(np.uint8)
    
        for i in range(0, self.model_regression_trees_matrix.shape[0]):
            regression_tree_vector = self.model_regression_trees_matrix[i]
            avarage_resiudal_leaf_vector = self.model_avarage_resiudal_leaf_matrix[i]
            cascade_contribution = cascade_contribution + self.model_learning_rate * predict_avarage_residual_vector_for_image(regression_tree_vector, avarage_resiudal_leaf_vector, I_intensities)
    
        S_hat_new = S_hat + list(zip(cascade_contribution[x_mask], cascade_contribution[y_mask]))

        features_hat_new = transformation_between_cascades(S_mean, S_hat_new, features_mean)
    
        return S_hat_new, features_hat_new
