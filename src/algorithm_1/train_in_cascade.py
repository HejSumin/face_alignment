from src.tree.tree_fitting import *

_DEBUG = True

"""
Hyperparameters

Parameters
    ----------
    _LEARNING_RATE : how much does the result of each tree influence the overall result
    _K : amount of trees per cascade
    _T : amount of cascades 
"""
_LEARNING_RATE = 0.1
_K = 500
_T = 10

def train_multiple_cascades(I_grayscale_matrix, S_hat_matrix, S_delta_matrix, S_true_matrix):
    run_output_logs = None
    if _DEBUG:
        run_output_logs = open('run_output/run_output_logs.txt', 'a', encoding='utf-8')
        log_str = "--------------------- new run ---------------------"
        print(log_str)
        run_output_logs.write(log_str + "\n")
        run_output_logs.flush()

    for t in range(0, _T):
        r_t_matrix = train_single_cascade(I_grayscale_matrix, S_delta_matrix)
        S_hat_matrix = S_hat_matrix + r_t_matrix
        S_delta_matrix = S_true_matrix - S_hat_matrix

        if _DEBUG:
            log_str = '-- [Succesfully generated cascade \t\t %d \t\t of a total of \t\t %d \t\t cascades] --' % (t+1, _T)
            print(log_str)
            run_output_logs.write(log_str + "\n")
            run_output_logs.flush()

    run_output_logs.close()
    return S_hat_matrix

def train_single_cascade(I_grayscale_matrix, S_delta_matrix):
    run_output_logs = None
    if _DEBUG:
        run_output_logs = open('run_output/run_output_logs.txt', 'a', encoding='utf-8')
        log_str = "--------------------- new cascade run ---------------------"
        print(log_str)
        run_output_logs.write(log_str + "\n")
        run_output_logs.flush()

    f_0_matrix = calculate_f_0_matrix(S_delta_matrix)
    f_k_minus_1_matrix = f_0_matrix

    for k in range(0, _K):
        r_i_k_matrix = calculate_residuals_matrix(S_delta_matrix, f_k_minus_1_matrix)
        regression_tree = generate_regression_tree(I_grayscale_matrix, r_i_k_matrix) #TODO save regression trees
        f_k_matrix = update_f_k_matrix(regression_tree, f_k_minus_1_matrix)
        f_k_minus_1_matrix = f_k_matrix

        if _DEBUG:
            log_str = '[Succesfully generated tree \t\t %d \t\t of a total of \t\t %d \t\t trees]' % (k+1, _K)
            print(log_str)
            run_output_logs.write(log_str + "\n")
            run_output_logs.flush()

    return f_k_minus_1_matrix

def calculate_residuals_matrix(S_delta_matrix, f_k_minus_1_matrix):
    return S_delta_matrix - f_k_minus_1_matrix

def calculate_f_0_matrix(S_delta_matrix):
    return np.mean(S_delta_matrix, axis=0) # TODO check if correct to use the mean

def update_f_k_matrix(regression_tree, f_k_minus_1_matrix, learning_rate=_LEARNING_RATE):
    g_k_matrix = regression_tree.get_avarage_residuals_matrix()
    f_k_matrix = f_k_minus_1_matrix + learning_rate * g_k_matrix
    return f_k_matrix
