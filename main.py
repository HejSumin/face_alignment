from timeit import default_timer as timer
from datetime import timedelta
import src as fa
import numpy as np

# images = 2000
# landmarks = 194
# R = 20
# n_image_matrix = np.random.randint(0, 256, (images, 400))
# I_grayscale_matrix = np.repeat(n_image_matrix, repeats=R, axis=0) # shape (N=n*R, #extraced pixels)
# S_delta_matrix = np.random.randint(1, 10, (R*images, landmarks)) * np.random.rand(R*images, landmarks) # 20 = R , images = amount of actual Images I

training_data = fa.create_training_triplets()
I_grayscale_matrix = training_data[:,0]
S_hat_matrix = training_data[:,1]
S_delta_matrix = training_data[:,2]
print("I_grayscale_matrix", I_grayscale_matrix.shape)
print("S_hat_matrix", S_hat_matrix.shape)
print("S_delta_matrix", S_delta_matrix.shape)

start = timer()
r_t_matrix = fa.build_regression_trees(I_grayscale_matrix, S_delta_matrix) 
end = timer()
print(r_t_matrix)
print("Time: ", timedelta(seconds=end-start))

run_output_results = open('run_output_results.txt', 'a', encoding='utf-8')
run_output_results.write(str(r_t_matrix) +"\n")
run_output_results.write("Time: " + str(timedelta(seconds=end-start)))
run_output_results.close()

# TODO connect to triplets
# TODO remove random matrix calculation