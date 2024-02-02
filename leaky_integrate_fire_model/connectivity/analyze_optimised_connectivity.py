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

str_identifier = "saved_data_optimised_connectivity/40"  # 25  # 26
fixed_hyperparameters_name = "/fixed_hyperparameters.json"
save_connectivity = True
try:
    with open(str_identifier + fixed_hyperparameters_name, "r") as configfile:
        json_parameters = json.load(configfile)
    # int_random_generator = 2
    int_random_generator = json_parameters['int_random_generator']
    N_total_nodes = json_parameters['N_total_nodes']
    weight = json_parameters['weight']
    option = json_parameters['option']
    number_of_seeds = json_parameters['number_of_seeds']
    if "dimensions" in json_parameters:
        dimensions = json_parameters['dimensions']
        pc = json_parameters['pc']
        dim_pc_fixed_parameter = True
    else:
        dim_pc_fixed_parameter = False
        dimensions = None
        pc = None
except IOError:
    print("Could not process the file due to IOError")
    sys.exit(1)
except KeyError:
    print("A key was not defined")
    sys.exit(2)

fixed_hyperparameters = {
    "N_total_nodes": N_total_nodes,
    "int_random_generator": int_random_generator,
    "weight": weight,
    "dimensions": dimensions,
    "pc": pc,
    "option": option,
    "number_of_seeds": number_of_seeds,
}
in_degree_elements, out_degree_elements = get_neuprint_data(option)

save_optimised_parameters = "optimised_parameters/" + option

file_name = str_identifier + "/top3_results.pkl"
png_extension = ".png"
pdf_extension = ".pdf"
hist_degree_distributions_name = "/hist_degree_distributions"
detailed_name = "_detailed"
top3_results = pd.read_pickle(file_name)
pd.set_option('display.max_columns', None)

str_name_dict = {
    "str_identifier": str_identifier,
    "hist_degree_distributions_name": hist_degree_distributions_name,
    "png_extension": png_extension,
    "pdf_extension": pdf_extension,
    "detailed_name": detailed_name
}

# reproduce_top3_results(top3_results, in_degree_elements, out_degree_elements, str_name_dict, fixed_hyperparameters,
#                            dim_pc_fixed_parameter, save_connectivity)

save_parameters_top_result(save_optimised_parameters, top3_results, fixed_hyperparameters, dim_pc_fixed_parameter)

plt.show()
