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
import networkx as nx

# option = "manc"
# option = "hemibrain"
option = "pyc-pyc"
N_total_nodes = 82
weight = 0.5
epsilon_connection_probabilities = [0.01, 0.05, 0.1]
# m_values = [50, 100, 150, 200]
m_values = [4, 7, 9]
int_for_random_generator = 1
log_type = False
binsize = 1

folder_name = "saved_data_reference_connectivity/"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

folder_identifier = 0
while os.path.exists(folder_name + "/" + str(folder_identifier)):
    folder_identifier += 1
str_identifier = folder_name + "/" + str(folder_identifier)
os.makedirs(str_identifier)


in_degree_elements, out_degree_elements = get_degree_data(option)

rng = np.random.default_rng(int_for_random_generator)

connectivity_matrix_er = [constant_probability_random_connectivity_matrix(N_total_nodes, epsilon_connection_probability, rng) for epsilon_connection_probability in epsilon_connection_probabilities]

connectivity_matrix_tri = np.tril(np.ones((N_total_nodes, N_total_nodes), dtype=int), k=-1)

connectivity_matrix_all = np.ones((N_total_nodes, N_total_nodes), dtype=int)

connectivity_matrix_none = np.zeros((N_total_nodes, N_total_nodes), dtype=int)

connectivity_matrix_single = np.diag(np.ones(N_total_nodes, dtype=int), k=-1)

# connectivity_matrix_ba = [simple_ba_graph(N_total_nodes, m_val, m_val, rng, a_arbitrary_constant=1000).astype(int) for m_val in m_values]

G_s = [nx.barabasi_albert_graph(n=N_total_nodes, m=m_val, seed=int_for_random_generator) for m_val in m_values]
connectivity_matrix_nx_ba = [nx.to_numpy_array(G, dtype=int) for G in G_s]

type_matrix_dict = {
    "er": connectivity_matrix_er,
    "tri": connectivity_matrix_tri,
    "all": connectivity_matrix_all,
    "none": connectivity_matrix_none,
    "single": connectivity_matrix_single,
    "nx ba": connectivity_matrix_nx_ba,
    # "ba": connectivity_matrix_ba
}
type_values_dict = {
    "er": epsilon_connection_probabilities,
    "tri": None,
    "all": None,
    "none": None,
    "single": None,
    "nx ba": m_values,
    # "ba": m_values
}


scores_dict = {}

for label, connectivity_matrix in type_matrix_dict.items():
    type_values = type_values_dict.get(label)
    figure, scores = score_and_plot(connectivity_matrix, label, in_degree_elements, out_degree_elements, option, weight, log_type=log_type, type_values=type_values, binsize=binsize)
    graph_type = label.replace(" ", "_")
    save_png_pdf(figure, str_identifier, "/hist_degree_distributions_" + graph_type)
    if isinstance(connectivity_matrix, np.ndarray):
        with open(str_identifier + "/connectivity_graph_" + graph_type + ".npy", "wb") as connectivity_file:
            np.save(connectivity_file, connectivity_matrix)
    elif isinstance(connectivity_matrix, list) and all(isinstance(matrix, np.ndarray) for matrix in connectivity_matrix):
        for count, (matrix, type_value) in enumerate(zip(connectivity_matrix, type_values)):
            with open(str_identifier + "/connectivity_graph_" + graph_type + str(type_value) + ".npy", "wb") as connectivity_file:
                np.save(connectivity_file, matrix)
    else:
        raise TypeError("connectivity_matrix should either be a np.ndarray or a list of np.ndarray ")
    scores_dict[label] = scores

with open(str_identifier + "/scores.json", "w") as scores_file:
    json.dump(scores_dict, scores_file)

options_dict = {
    "option": option,
    "N_total_nodes": N_total_nodes,
    "weight": weight,
    "int_for_random_generator": int_for_random_generator
}
with open(str_identifier + "/options.json", "w") as options_file:
    json.dump(options_dict, options_file)

plt.show()
