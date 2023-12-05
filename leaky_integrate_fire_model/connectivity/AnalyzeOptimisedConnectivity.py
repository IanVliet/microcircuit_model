import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
from FunctionsExpandedBrunelNetwork import *
from IPython.display import display
import scipy.io
import os
import json
import sys

str_identifier = "saved_data_optimised_connectivity/6"
fixed_hyperparameters_name = "/fixed_hyperparameters.json"

try:
    with open(str_identifier + fixed_hyperparameters_name, "r") as configfile:
        json_parameters = json.load(configfile)
    int_random_generator = json_parameters['int_random_generator']
    N_total_nodes = json_parameters['N_total_nodes']
    weight = json_parameters['weight']
    option = json_parameters['option']
except IOError:
    print("Could not process the file due to IOError")
    sys.exit(1)
except KeyError:
    print("A key was not defined")
    sys.exit(2)

if option == "manc":
    with open('../manc_v1.0_indegrees.npy', 'rb') as indegree_file:
        in_degree_elements = np.load(indegree_file)

    with open('../manc_v1.0_outdegrees.npy', 'rb') as outdegree_file:
        out_degree_elements = np.load(outdegree_file)
elif option == "hemibrain":
    with open('../hemibrain_v1.2.1_indegrees.npy', 'rb') as indegree_file:
        in_degree_elements = np.load(indegree_file)

    with open('../hemibrain_v1.2.1_outdegrees.npy', 'rb') as outdegree_file:
        out_degree_elements = np.load(outdegree_file)
else:
    print("The chosen option", option, "is not one of the possible options.")
    exit(1)


file_name = str_identifier + "/top3_results.pkl"
png_extension = ".png"
pdf_extension = ".pdf"
hist_degree_distributions_name = "/hist_degree_distributions"
top3_results = pd.read_pickle(file_name)
pd.set_option('display.max_columns', None)

for index_result, result in top3_results.iterrows():
    rng = np.random.default_rng(int_random_generator)
    m_0_nodes = result["config/m_0"]
    rho_probability = result["config/rho"]  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions = result["config/l"]  # Likely L in the web application
    E_k = result["config/E_k"]  # likely EK in the web application
    phi_U_probability = result["config/phi_U"]  # likely phi_U in the web application
    phi_D_probability = result["config/phi_D"]  # likely phi_D in the web application
    delta = result["config/delta"]
    noise_factor = result["config/noise_factor"]
    dimensions = result["config/dimensions"]
    pc = result["config/pc"]
    graph = convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions,
                                  N_total_nodes, E_k, phi_U_probability,
                                  phi_D_probability, rng, in_degree_elements, out_degree_elements, dimensions,
                                  delta=delta, noise_factor=noise_factor, spatial=True, pc=pc)
    # save graph
    if not os.path.exists(str_identifier):
        os.makedirs(str_identifier)

    folder_identifier = 0
    while os.path.exists(str_identifier + "/results_data/" + str(folder_identifier)):
        folder_identifier += 1
    data_directory = str_identifier + "/results_data/" + str(folder_identifier)
    os.makedirs(data_directory)

    with open(data_directory + "/connectivity_graph.npy", "wb") as connectivityfile:
        np.save(connectivityfile, graph)

    # results from generative convolutional model
    in_degrees, in_degree_counts = np.unique(np.sum(graph, axis=1), return_counts=True)
    out_degrees, out_degree_counts = np.unique(np.sum(graph, axis=0), return_counts=True)
    in_degree_elements_model = np.repeat(in_degrees, in_degree_counts)
    out_degree_elements_model = np.repeat(out_degrees, out_degree_counts)

    in_degree_elements_matlab_sim = out_degree_elements_matlab_sim = [0]

    config = {
        "m_0_nodes": m_0_nodes,
        "rho_probability": rho_probability,
        "l_cardinality_partitions": l_cardinality_partitions,
        "E_k": E_k,
        "phi_U_probability": phi_U_probability,
        "phi_D_probability": phi_D_probability,
        "delta": delta,
        "noise_factor": noise_factor,
        "dimensions": dimensions,
        "pc": pc,
    }
    print(config)
    figure = hist_plot_data_model_degree_distributions(option, in_degree_elements, in_degree_elements_model,
                                                       in_degree_elements_matlab_sim, out_degree_elements,
                                                       out_degree_elements_model,
                                                       out_degree_elements_matlab_sim)
    figure.savefig(data_directory + hist_degree_distributions_name + png_extension)
    figure.savefig(data_directory + hist_degree_distributions_name + pdf_extension)

    # degree distributions of data
    in_degree_values, in_degree_distribution, out_degree_values, out_degree_distribution = \
        get_degree_distributions(in_degree_elements, out_degree_elements)

    in_degree, out_degree = \
        get_interpolated_degree_distributions(in_degree_values, in_degree_distribution, out_degree_values,
                                              out_degree_distribution, np.max(in_degree_elements),
                                              np.max(out_degree_elements))
    # degree distributions of model

    in_degree_values_model, in_degree_distribution_model, out_degree_values_model, out_degree_distribution_model = \
        get_degree_distributions(in_degree_elements_model, out_degree_elements_model)

    in_degree_gen_model, out_degree_gen_model = \
        get_interpolated_degree_distributions(in_degree_values_model, in_degree_distribution_model,
                                              out_degree_values_model,
                                              out_degree_distribution_model, np.max(in_degree_elements_model),
                                              np.max(out_degree_elements_model))
    if len(in_degree) > len(in_degree_gen_model):
        in_degree_gen_model_extended = extend_with_zeros(len(in_degree), in_degree_gen_model) + 1e-15
    else:
        in_degree_gen_model_extended = in_degree_gen_model + 1e-15
    if len(out_degree) > len(out_degree_gen_model):
        out_degree_gen_model_extended = extend_with_zeros(len(out_degree), out_degree_gen_model) + 1e-15
    else:
        out_degree_gen_model_extended = out_degree_gen_model + 1e-15

    in_degree_cross_entropy = cross_entropy(in_degree, in_degree_gen_model_extended)
    out_degree_cross_entropy = cross_entropy(out_degree, out_degree_gen_model_extended)
    print("original score", result["score"])
    print("recreated score:", linear_combination_cross_entropy(in_degree_cross_entropy, out_degree_cross_entropy, weight=weight))

plt.show()
