from timeit import default_timer as timer
from datetime import timedelta
import src as fa
import numpy as np

# Paths
data_path = 'data/'
annotations_path = 'data/annotation/'

print("... starting training ...")
# print("... creating training data ... ")
# I_grayscale_matrix, S_hat_matrix, S_delta_matrix, S_true_matrix = fa.create_training_triplets(train_images_path=data_path+'train_1/')
# # np.save("np_data/run_input_I_grayscale_matrix", I_grayscale_matrix)
# # np.save("np_data/run_input_S_hat_matrix", S_hat_matrix)
# # np.save("np_data/run_input_S_delta_matrix", S_delta_matrix)
# # np.save("np_data/run_input_S_true_matrix", S_true_matrix)

print("... loading training data ... ")
I_grayscale_matrix = np.load("np_data/run_input_I_grayscale_matrix.npy")
S_hat_matrix = np.load("np_data/run_input_S_hat_matrix.npy")
S_delta_matrix = np.load("np_data/run_input_S_delta_matrix.npy")
S_true_matrix = np.load("np_data/run_input_S_true_matrix.npy")

print("... finished loading training data ...")
print("... starting training in cascade ...")
start = timer()
S_hat_matrix = fa.train_multiple_cascades(I_grayscale_matrix, S_hat_matrix, S_delta_matrix, S_true_matrix) 
end = timer()
print(S_hat_matrix)

np.save("run_output/run_output_numpy_S_hat_matrix", S_hat_matrix)

print("Run finished in: (Time)", timedelta(seconds=end-start))

run_output_results = open('run_output_results.txt', 'a', encoding='utf-8')
run_output_results.write("---------------------run---------------------\n")
run_output_results.write("Run finished in: (Time)" + str(timedelta(seconds=end-start)) + "\n")
run_output_results.close()
