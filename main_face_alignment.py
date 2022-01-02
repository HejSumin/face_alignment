from timeit import default_timer as timer
from datetime import timedelta

from numpy.core.arrayprint import dtype_is_implied
import src as fa
import numpy as np
import os

if os.name == 'posix':
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

# Paths
data_path = 'data/'
annotations_path = 'data/annotation/'

# print("... starting training ...")
# print("... creating training data ... ")
# training_data = fa.create_training_data(data_path + "train_1/", annotations_path)
# # np.save("np_data/run_input_training_data", training_data)

# print("... loading training data ... ")
# training_data = np.load("np_data/run_input_training_data.npy", allow_pickle=True)

# #Not working, index out of bound when loading new intensities
# print("... starting training trees 🌳 in cascade ...")
# start = timer()
# training_data_result = fa.train_multiple_cascades(training_data, use_exponential_prior=True)
# end = timer()

# np.save("run_output/run_output_numpy_training_data_result", training_data_result)

# print("Run finished in: (Time)", timedelta(seconds=end-start))


training_data = np.load("np_data/run_input_training_data.npy", allow_pickle=True)
I_intensities = training_data[0,3]

regression_trees_matrix = np.load("run_output/model_regression_trees_matrix_cascade_2.npy")
avarage_residual_leaf_matrix = np.load("run_output/model_avarage_residual_leaf_matrix_cascade_2.npy")

model_regression_trees = np.load("run_output/model_regression_trees_object_cascade_2.npy", allow_pickle=True)

predict_nice = fa.predict_avarage_residual_vector_for_image(regression_trees_matrix[2], avarage_residual_leaf_matrix[2], I_intensities)
predict_not_nice = fa.predict_avarage_residual_vector_for_image_from_regression_tree_object(model_regression_trees[2], I_intensities)

print(predict_nice - predict_not_nice)