from timeit import default_timer as timer
from datetime import timedelta
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
# np.save("np_data/run_input_training_data", training_data)

print("... loading training data ... ")
training_data = np.load("np_data/run_input_training_data.npy", allow_pickle=True)

#Not working, index out of bound when loading new intensities
print("... starting training trees 🌳 in cascade ...")
start = timer()
training_data_result = fa.train_multiple_cascades(training_data, use_exponential_prior=True)
end = timer()

np.save("run_output/run_output_numpy_training_data_result", training_data_result)

#print("Run finished in: (Time)", timedelta(seconds=end-start))
