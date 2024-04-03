import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
from connectivity_utils import *
from IPython.display import display
import scipy.io
import os
import json
import sys

identifiers = [7, 9]
base_folder = "saved_data_optimised_connectivity/"
str_identifiers = [base_folder + str(identifier) for identifier in identifiers]

# str_identifier = "saved_data_optimised_connectivity/5"  # 43  # 44  # 25  # 26
fixed_hyperparameters_name = "/fixed_hyperparameters.json"
save_connectivity = False
log_type = False
single_figure = True
binsize = 1


network_sizes = [4600, 9200, 13800, 18400, 23000]
# network_sizes = [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12500, 15000, 17500, 20000]
# network_sizes = [6, 10, 20, 30, 40, 50, 82, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

# create figure and axis for plotting cross-entropy vs network size
fig_and_ax = plt.subplots()

for identifier_index, str_identifier in enumerate(str_identifiers):
    json_parameters, fixed_hyperparameters, dim_pc_fixed_parameter = get_fixed_hyperparameters(str_identifier, fixed_hyperparameters_name, number_of_seeds=10)
    option = fixed_hyperparameters["option"]
    N_total_nodes = fixed_hyperparameters["N_total_nodes"]
    weight = fixed_hyperparameters["weight"]

    in_degree_elements, out_degree_elements = get_degree_data(option)
    save_optimised_parameters, file_name, str_name_dict = (
        get_standard_file_and_folder_names(option, str_identifier, fixed_hyperparameters_name, log_type, N_total_nodes))

    top3_results = pd.read_pickle(file_name)
    pd.set_option('display.max_columns', None)

    # configurations for full analysis:
    # the number of nodes, and for each the binsize and plot-type (either 'log' or 'lin').
    # total_nodes_binsize_plot_type_dict = {
    #     json_parameters['N_total_nodes']: [[1, 'log'], [5, 'log'], [5, 'lin'], [10, 'log'], [10, 'lin'],
    #                                        [20, 'log'], [20, 'lin'], [50, 'log'], [50, 'lin']],
    #     10000: [[1, 'log'], [20, 'log']],
    #     5000: [[1, 'log'], [20, 'log']],
    # }
    total_nodes_binsize_plot_type_dict = {
        json_parameters['N_total_nodes']: [[1, 'log'], [5, 'lin'], [10, 'log']],
    }

    # reproduce_top3_results(top3_results, in_degree_elements, out_degree_elements, str_name_dict, fixed_hyperparameters,
    #                        dim_pc_fixed_parameter, save_connectivity, one_seed_only=False, log_type=log_type,
    #                        binsize=binsize, single_figure=single_figure)

    # full_analysis_optimised_connectivity(top3_results, in_degree_elements, out_degree_elements, str_name_dict, fixed_hyperparameters,
    #                            dim_pc_fixed_parameter, save_connectivity=False, total_nodes_binsize_plot_type_dict=total_nodes_binsize_plot_type_dict)

    # save_parameters_top_result(save_optimised_parameters, top3_results, fixed_hyperparameters, dim_pc_fixed_parameter)

    # calculates entropy for the distribution of the dataset (instead of the model indegree and outdegree elements,
    # it is only the indegree and outdegree elements of the dataset)
    entropy_score = elements_linear_cross_entropy(in_degree_elements, out_degree_elements, in_degree_elements,
                                                    out_degree_elements, weight)
    print(entropy_score)
    save_plot = (identifier_index == len(str_identifiers) - 1)
    plot_cross_entropy_vs_network_size(top3_results.iloc[0], in_degree_elements, out_degree_elements, fixed_hyperparameters,
                                       str_name_dict, dim_pc_fixed_parameter, network_sizes=network_sizes,
                                       save_connectivity=False, log_type=log_type, fig_and_ax=fig_and_ax, save_plot=save_plot)


plt.show()
