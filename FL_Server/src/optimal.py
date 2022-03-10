import numpy as np


# TODO: Run optimal algorithm
def run_optimal(num_clients):
    f_n = [i for i in range(num_clients)]
    v_n = [i * 100 for i in range(1, num_clients + 1)]

    return [f_n, v_n]


def get_optimal_value(num_clients, opt_name):
    opt_value = run_optimal(num_clients)
    if len(opt_value) != len(opt_name):
        print("Check the length opt_name and opt_value")

    return [{opt_name[i]: opt_value[i][j] for i in range(len(opt_name))} for j in range(num_clients)]
