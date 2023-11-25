import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
from FunctionsExpandedBrunelNetwork import *
import scipy.io
import os

# option = "C_elegans"
# option = "matlab_data"
option = "manc"
# option = "other"
if option == "C_elegans":
    in_elements_elegans = {}
    out_elements_elegans = {}
    with open("../large-scale models/brain connectivity simulator/connections_list_Celegans.txt") as elegans_connections:
        for connection_pair in elegans_connections:
            first_el, sec_element = [el for el in connection_pair.split(',')]
            if first_el not in out_elements_elegans:
                out_elements_elegans[first_el] = 1
            else:
                out_elements_elegans[first_el] += 1
            if sec_element not in in_elements_elegans:
                in_elements_elegans[sec_element] = 1
            else:
                in_elements_elegans[sec_element] += 1
    in_degree_elements = np.array(list(in_elements_elegans.values()))
    out_degree_elements = np.array(list(out_elements_elegans.values()))
    m_0_nodes_choice = 25  # equivalent to "m" in the web application?
    rho_probability_choice = 0.1  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions_choice = 5  # Likely L in the web application
    N_total_nodes_choice = 10000  # likely N in the web application
    E_k_choice = 2  # likely EK in the web application
    phi_U_probability_choice = 0.9  # likely phi_U in the web application
    phi_D_probability_choice = 0.0005  # likely phi_D in the web application
    delta = 1.5
    noise_factor = 3
    dimensions = [500, 0, 500, 0, 2000, 0]
elif option == "matlab_data":
    mat = scipy.io.loadmat("../Convolutive model/261881/Giacopelli_et_al_2020_ModelDB/data.mat")
    in_degree_elements, out_degree_elements = mat['dinEE'][0], mat['doutEE'][0]
    in_degree_elements = np.array(in_degree_elements)
    out_degree_elements = np.array(out_degree_elements)
    sim_matlab = scipy.io.loadmat("../Convolutive model/261881/Giacopelli_et_al_2020_ModelDB/sim_data.mat")
    in_degree_elements_matlab_sim, out_degree_elements_matlab_sim = sim_matlab['sim_dinEE'][0], sim_matlab['sim_doutEE'][0]
    m_0_nodes_choice = 100  # equivalent to "m" in the web application?
    rho_probability_choice = 0.1  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions_choice = 100  # Likely L in the web application
    N_total_nodes_choice = 10000  # likely N in the web application
    E_k_choice = 115  # likely EK in the web application
    phi_U_probability_choice = 0.9  # likely phi_U in the web application
    phi_D_probability_choice = 0.0005  # likely phi_D in the web application
    delta = 0.1
    noise_factor = 3
    dimensions = [100, 0, 100, 0, 100, 0]
    print(option)
elif option == "manc":
    with open('manc_v1.0_indegrees.npy', 'rb') as indegree_file:
        in_degree_elements = np.load(indegree_file)

    with open('manc_v1.0_outdegrees.npy', 'rb') as outdegree_file:
        out_degree_elements = np.load(outdegree_file)
    m_0_nodes_choice = 100  # equivalent to "m" in the web application?
    rho_probability_choice = 0.1  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions_choice = 20  # Likely L in the web application. N_total_nodes/2 should be divisible by l
    N_total_nodes_choice = 10000  # likely N in the web application
    E_k_choice = 150  # likely EK in the web application
    phi_U_probability_choice = 0.9  # likely phi_U in the web application
    phi_D_probability_choice = 0.005  # likely phi_D in the web application
    delta = 0.1
    noise_factor = 0.3
    dimensions = [1, 0, 1, 0, 1, 0]
else:
    in_degree_elements, out_degree_elements = \
        get_csv_degree_elements("connectome layer 4 of the somatosensory cortex indegree.csv",
                                     "connectome layer 4 of the somatosensory cortex outdegree.csv")
    # data from "https://l4dense2019.brain.mpg.de/webdav/" (connectome.csv). However, is not suited for eventual use in
    # simulation. The proximal dendrites, smooth dendrites, etc. are not (yet) assigned to cells within the dataset.
    # Therefore, there are connections present in the data which do not connect to cells in the dataset. This causes
    # a seeming need of more connections between the modelled cells which does not represent the network underlying the data
    m_0_nodes_choice = 100  # equivalent to "m" in the web application?
    rho_probability_choice = 0.1  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions_choice = 25  # Likely L in the web application
    N_total_nodes_choice = 10000  # likely N in the web application
    E_k_choice = 1  # likely EK in the web application
    phi_U_probability_choice = 1  # likely phi_U in the web application
    phi_D_probability_choice = 0.0001  # likely phi_D in the web application
    delta = 1.5
    noise_factor = 3
    dimensions = [1, 0, 1, 0, 1, 0]

rng = np.random.default_rng(42)
# web application: https://gtalg.ebrains-italy.eu/connect/

start_generation = time.time()
B = convolutive_graph_gen(m_0_nodes_choice, rho_probability_choice, l_cardinality_partitions_choice,
                          N_total_nodes_choice, E_k_choice, phi_U_probability_choice,
                          phi_D_probability_choice, rng, in_degree_elements, out_degree_elements, dimensions, pc=300,
                          delta=delta, noise_factor=noise_factor, spatial=True)
# A = spatial_block_ba_graph(N_total_nodes_choice, 1, 0, 1, 0, 1, 0, 1.5, 3, 0, rng,
#                            in_degree_elements, out_degree_elements)[0]
end_generation = time.time()
print(str(end_generation-start_generation) + " s")

if not os.path.exists('saved_data_connectivity'):
    os.makedirs('saved_data_connectivity')

folder_identifier = 0
while os.path.exists("saved_data_connectivity/" + str(folder_identifier)):
    folder_identifier += 1
str_identifier = "saved_data_connectivity/" + str(folder_identifier)
os.makedirs(str_identifier)

with open(str_identifier + "/connectivity_graph.npy", "wb") as connectivityfile:
    np.save(connectivityfile, B)

plot_degree_counts(B)

# results from original data
in_degree_values, in_degree_distribution, out_degree_values, out_degree_distribution = \
    get_degree_distributions(in_degree_elements, out_degree_elements)

in_degree, out_degree = \
    get_interpolated_degree_distributions(in_degree_values, in_degree_distribution, out_degree_values,
                                          out_degree_distribution, np.max(in_degree_elements),
                                          np.max(out_degree_elements))

# results from generative convolutional model
in_degrees, in_degree_counts = np.unique(np.sum(B, axis=1), return_counts=True)
out_degrees, out_degree_counts = np.unique(np.sum(B, axis=0), return_counts=True)
in_degree_elements_model = np.repeat(in_degrees, in_degree_counts)
out_degree_elements_model = np.repeat(out_degrees, out_degree_counts)

fig, ax = plt.subplots(1, 2)
num_bins = len(np.unique(in_degree_elements))
density_type = True
if num_bins > 100:
    num_bins = 100
ax[0].hist(in_degree_elements, label='data', alpha=1, bins=num_bins, density=density_type, edgecolor='black', linewidth=1)
ax[0].hist(in_degree_elements_model, label='sim', alpha=0.8, bins=num_bins, density=density_type, edgecolor='red',
           linewidth=2, histtype='step')
if option == "matlab_data":
    ax[0].hist(in_degree_elements_matlab_sim, label='matlab sim', alpha=0.8, bins=num_bins, density=density_type,
               edgecolor='green', linewidth=2, histtype='step')
ax[0].set_xlabel("indegree")
ax[0].set_ylabel("probability")
ax[1].hist(out_degree_elements, label='data', alpha=1, bins=num_bins, density=density_type, edgecolor='black', linewidth=1)
ax[1].hist(out_degree_elements_model, label='sim', alpha=0.8, bins=num_bins, density=density_type, edgecolor='red',
           linewidth=2, histtype='step')
if option == "matlab_data":
    ax[1].hist(out_degree_elements_matlab_sim, label='matlab sim', alpha=0.8, bins=num_bins, density=density_type,
               edgecolor='green', linewidth=2, histtype='step')
ax[1].set_xlabel("outdegree")
ax[1].set_ylabel("probability")
ax[0].legend()
ax[1].legend()


in_degree_values_model, in_degree_distribution_model, out_degree_values_model, out_degree_distribution_model = \
    get_degree_distributions(in_degree_elements_model, out_degree_elements_model)

in_degree_gen_model, out_degree_gen_model = \
    get_interpolated_degree_distributions(in_degree_values_model, in_degree_distribution_model, out_degree_values_model,
                                          out_degree_distribution_model, np.max(in_degrees), np.max(out_degrees))

# in_degree_model, out_degree_model = \
#     convolutive_probabilities(in_degree, out_degree, N_total_nodes_choice, l_cardinality_partitions_choice,
#                               p_probability_choice, phi_U_probability_choice, phi_D_probability_choice)

# results from online simulator
# indegrees:
if option == "C_elegans":
    in_degree_online_values = []
    in_degree_online_probs = []
    with open("../large-scale models/brain connectivity simulator/"
              "indegree_distribution_Celegans_online.txt") as in_degrees_online:
        for line in in_degrees_online:
            in_degree_amount, in_degree_amount_prob = [value for value in line.split(',')]
            in_degree_online_values.append(in_degree_amount)
            in_degree_online_probs.append(float(in_degree_amount_prob))
    in_degree_online_probs = np.array(in_degree_online_probs)
    # outdegrees:
    out_degree_online_values = []
    out_degree_online_probs = []
    with open("../large-scale models/brain connectivity simulator/"
              "outdegree_distribution_Celegans_online.txt") as out_degrees_online:
        for line in out_degrees_online:
            out_degree_amount, out_degree_amount_prob = [value for value in line.split(',')]
            out_degree_online_values.append(out_degree_amount)
            out_degree_online_probs.append(float(out_degree_amount_prob))

    out_degree_online_values = np.array(out_degree_online_values)
    out_degree_online_probs = np.array(out_degree_online_probs)


fig1, ax1 = plt.subplots()
ax1.plot(in_degree, label="data")
ax1.plot(in_degree_gen_model, label="own generative model")
if option == "C_elegans":
    ax1.plot(in_degree_online_probs, label="online simulator")
ax1.legend()
ax1.set_xlabel("indegree")
ax1.set_ylabel("probability")

fig2, ax2 = plt.subplots()
ax2.plot(out_degree, label="data")
ax2.plot(out_degree_gen_model, label="own generative model")
if option == "C_elegans":
    ax2.plot(out_degree_online_probs, label="online simulator")
# ax1.plot(in_degree_model, label="in-degree model")
# ax2.plot(out_degree_model, label="out-degree model")
ax2.legend()
ax2.set_xlabel("outdegree")
ax2.set_ylabel("probability")

plt.show()
