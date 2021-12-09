from timeit import default_timer as timer
from datetime import timedelta
import src as fa
import numpy as np

# Paths
data_path = 'data/'
annotations_path = 'data/annotation/'

I_grayscale_matrix, S_hat_matrix, S_delta_matrix = fa.create_training_triplets(train_images_path=data_path+'train_1/')
# print("I_grayscale_matrix", I_grayscale_matrix.shape)
# print("S_hat_matrix", S_hat_matrix.shape)
# print("S_delta_matrix", S_delta_matrix.shape)

np.save("run_input_I_grayscale_matrix", I_grayscale_matrix)
np.save("run_input_S_hat_matrix", S_hat_matrix)
np.save("run_input_S_delta_matrix", S_delta_matrix)

start = timer()
r_t_matrix = fa.build_regression_trees(I_grayscale_matrix, S_delta_matrix) 
end = timer()
print(r_t_matrix)

np.save("run_output_numpy_r_t_matrix", r_t_matrix)

print("Time: ", timedelta(seconds=end-start))

run_output_results = open('run_output_results.txt', 'a', encoding='utf-8')
run_output_results.write(str(r_t_matrix) +"\n")
run_output_results.write("Time: " + str(timedelta(seconds=end-start)))
run_output_results.close()
