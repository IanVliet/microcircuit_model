import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
from connectivity_utils import *
from data_preparation.data_prep_utils import *
import scipy.io
import os

# option = "C_elegans"
# option = "matlab_data"
option = "manc"
# option = "other"

in_degree_elements_matlab_sim = out_degree_elements_matlab_sim = [0]

if option == "C_elegans":
    in_elements_elegans = {}
    out_elements_elegans = {}
    with open("../../large-scale models/brain connectivity simulator/connections_list_Celegans.txt") as elegans_connections:
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
    pc = 300
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
    pc = 300
    print(option)
elif option == "manc":
    with open('data_preparation/manc_v1.0_indegrees.npy', 'rb') as indegree_file:
        in_degree_elements = np.load(indegree_file)

    with open('data_preparation/manc_v1.0_outdegrees.npy', 'rb') as outdegree_file:
        out_degree_elements = np.load(outdegree_file)
    m_0_nodes_choice = 72  # equivalent to "m" in the web application?
    rho_probability_choice = 0.1  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions_choice = 10  # Likely L in the web application. N_total_nodes/2 should be divisible by l
    N_total_nodes_choice = 10000  # likely N in the web application
    E_k_choice = -44.81936312325523  # likely EK in the web application
    phi_U_probability_choice = 0.9627767039933294  # likely phi_U in the web application
    phi_D_probability_choice = 0.0032519618269504303  # likely phi_D in the web application
    delta = 0.5285813187252297
    noise_factor = 0.23553714358192923
    dimensions = [1, 0, 1, 0, 1, 0]
    pc = 300
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
    pc = 300

rng = np.random.default_rng(42)
# web application: https://gtalg.ebrains-italy.eu/connect/

start_generation = time.time()
B = convolutive_graph_gen(m_0_nodes_choice, rho_probability_choice, l_cardinality_partitions_choice,
                          N_total_nodes_choice, E_k_choice, phi_U_probability_choice,
                          phi_D_probability_choice, rng, in_degree_elements, out_degree_elements, dimensions, pc=pc,
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

hist_plot_data_model_degree_distributions(option, in_degree_elements, in_degree_elements_model,
                                          in_degree_elements_matlab_sim, out_degree_elements, out_degree_elements_model,
                                          out_degree_elements_matlab_sim)

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
in_degree_online_probs = []
out_degree_online_probs = []
if option == "C_elegans":
    in_degree_online_values = []
    with open("../../large-scale models/brain connectivity simulator/indegree_distribution_Celegans_online.txt") as in_degrees_online:
        for line in in_degrees_online:
            in_degree_amount, in_degree_amount_prob = [value for value in line.split(',')]
            in_degree_online_values.append(in_degree_amount)
            in_degree_online_probs.append(float(in_degree_amount_prob))
    in_degree_online_probs = np.array(in_degree_online_probs)
    # outdegrees:
    out_degree_online_values = []
    with open("../../large-scale models/brain connectivity simulator/outdegree_distribution_Celegans_online.txt") as out_degrees_online:
        for line in out_degrees_online:
            out_degree_amount, out_degree_amount_prob = [value for value in line.split(',')]
            out_degree_online_values.append(out_degree_amount)
            out_degree_online_probs.append(float(out_degree_amount_prob))

    out_degree_online_values = np.array(out_degree_online_values)
    out_degree_online_probs = np.array(out_degree_online_probs)

step_plot_data_model_degree_distributions(option, in_degree, in_degree_gen_model, in_degree_online_probs,
                                          out_degree, out_degree_gen_model, out_degree_online_probs)

plt.show()
