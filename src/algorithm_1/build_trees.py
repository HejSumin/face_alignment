from src.tree.tree_fitting import *

_LEARNING_RATE = 0.1
_K = 500

_DEBUG = True

def build_regression_trees(I_grayscale_matrix, S_delta_matrix):
    run_output_logs = None
    if _DEBUG:
        run_output_logs = open('run_output_logs.txt', 'a', encoding='utf-8')
        run_output_logs.write("\n--------------------- new run ---------------------\n")

    f_0_matrix = calculate_f_0_matrix(S_delta_matrix)
    f_k_minus_1_matrix = f_0_matrix

    for k in range(0, _K):

        r_i_k_matrix = calculate_residuals_matrix(S_delta_matrix, f_k_minus_1_matrix)
        regression_tree = generate_regression_tree(I_grayscale_matrix, r_i_k_matrix)
        f_k_matrix = update_f_k_matrix(regression_tree, f_k_minus_1_matrix)
        f_k_minus_1_matrix = f_k_matrix

        if _DEBUG:
            log_str = '[Succesfully generated tree \t %d \t of a total of \t %d \t trees]\n' % (k+1, _K)
            print(log_str)
            run_output_logs.write(log_str)
            if k % 10 == 0:
                log_str_f_k_minus_1_matrix = 'f_k_minus_1_matrix' + str(f_k_minus_1_matrix) + "\n"
                print(log_str_f_k_minus_1_matrix)
                run_output_logs.write(log_str_f_k_minus_1_matrix)

    run_output_logs.close()
    return f_k_minus_1_matrix

def calculate_residuals_matrix(S_delta_matrix, f_k_minus_1_matrix):
    return S_delta_matrix - f_k_minus_1_matrix

def calculate_f_0_matrix(S_delta_matrix):
    return np.mean(S_delta_matrix, axis=0) # TODO check if correct to use the mean

def update_f_k_matrix(regression_tree, f_k_minus_1_matrix, learning_rate=_LEARNING_RATE):
    g_k_matrix = regression_tree.get_avarage_residuals_matrix()
    f_k_matrix = f_k_minus_1_matrix + learning_rate * g_k_matrix
    return f_k_matrix
