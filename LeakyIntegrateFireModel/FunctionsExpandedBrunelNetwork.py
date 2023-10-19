import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
import csv
from collections import Counter
from scipy.special import comb


def poisson_distribution_spike_generation(freq, total_time, time_step_size, cells, connections, rng):
    # generate external spike trains
    average_number_of_spikes = ceil(freq * total_time)  # freq * time interval ([kHz]*[ms])
    average_interval_in_steps = 1/(freq * time_step_size)
    external_spike_intervals = rng.poisson(average_interval_in_steps,
                                                 size=(cells, connections, average_number_of_spikes))
    # uniform_spike_start = np.random.randint(0, round(average_interval_in_steps*2), (number_of_cells, C_ext_connections))
    # complete_external_spike_intervals = np.dstack((
    #         uniform_spike_start,
    #         external_spike_intervals
    #     ))
    # external_spike_times = np.cumsum(complete_external_spike_intervals, axis=2)
    external_spike_times = np.cumsum(external_spike_intervals, axis=2)
    # minimum_indices = []
    latest_spikes_smallest_value = np.min(external_spike_times[:, :, -1])
    while latest_spikes_smallest_value <= total_time / time_step_size:
        # minimum_indices.append(np.unravel_index(np.argmin(external_spike_times[:, :, -1])
        # , external_spike_times[:, :, -1].shape))
        rest_time = total_time - latest_spikes_smallest_value * time_step_size
        rest_expected_spikes = ceil(freq * rest_time)
        rest_external_spike_intervals = np.dstack((
            external_spike_times[:, :, -1],
            rng.poisson(average_interval_in_steps, size=(cells, connections, rest_expected_spikes))
        ))
        rest_external_spike_times = np.cumsum(rest_external_spike_intervals, axis=2)
        external_spike_times = np.dstack([
            external_spike_times[:, :, :-1], rest_external_spike_times
        ])
        latest_spikes_smallest_value = np.min(external_spike_times[:, :, -1])
    # for indices in minimum_indices:
        # print(external_spike_times[indices])
    return external_spike_times


def uniform_log_spike_generation(freq, total_time, time_step_size, total_time_steps, cells, connections, rng):
    # generate external spike trains
    average_number_of_spikes = ceil(freq * total_time)  # freq * time interval ([kHz]*[ms])
    uniform_for_external_spike_intervals = \
        rng.uniform(0, 1, (cells, connections, average_number_of_spikes))
    external_spike_intervals = -np.log(1 - uniform_for_external_spike_intervals) / (freq*time_step_size)
    external_spike_times = np.cumsum(external_spike_intervals, axis=2, dtype=int)
    # minimum_indices = []
    while np.min(external_spike_times[:, :, -1]) <= total_time_steps:
        # minimum_indices.append(np.unravel_index(np.argmin(external_spike_times[:, :, -1]),
        #                                         external_spike_times[:, :, -1].shape))
        rest_time = total_time - np.min(external_spike_times[:, :, -1]) * time_step_size
        rest_expected_spikes = ceil(freq * rest_time)
        rest_uniform_for_external_spike_intervals = \
            rng.uniform(0, 1, (cells, connections, rest_expected_spikes))
        rest_external_spike_intervals = np.dstack((
            external_spike_times[:, :, -1],
            -np.log(1 - rest_uniform_for_external_spike_intervals) / (freq*time_step_size)
        ))
        rest_external_spike_times = np.cumsum(rest_external_spike_intervals, axis=2)
        external_spike_times = np.block([
            external_spike_times[:, :, :-1], rest_external_spike_times
        ])
    # for indices in minimum_indices:
    #     print(external_spike_times[indices])
    external_spike_times = np.rint(external_spike_times)
    return external_spike_times


def uniform_probability_spike_generation(freq, total_time, time_step_size, total_time_steps, cells, connections, rng):
    # generate external spike trains
    uniform_for_external_spike_intervals = \
        rng.uniform(0, 1, (cells, connections, total_time_steps))
    external_spike_steps = uniform_for_external_spike_intervals <= freq*time_step_size
    return external_spike_steps


def constant_probability_random_connectivity_matrix(num_cells, probability, rng):
    connectivity_matrix = rng.uniform(size=(num_cells, num_cells)) <= probability
    np.fill_diagonal(connectivity_matrix, 0)
    return connectivity_matrix


def get_interpolated_degree_distributions(in_degree_elements, out_degree_elements):
    in_degree_values, in_degree_counts = np.unique(in_degree_elements, return_counts=True)
    out_degree_values, out_degree_counts = np.unique(out_degree_elements, return_counts=True)
    in_degree_distribution = in_degree_counts/np.sum(in_degree_counts)
    out_degree_distribution = out_degree_counts/np.sum(out_degree_counts)
    max_in_degree = int(np.max(in_degree_values))
    n_range_in_degree = np.arange(max_in_degree)
    max_out_degree = int(np.max(in_degree_values))
    n_range_out_degree = np.arange(max_out_degree)
    total_in_degree_distribution = np.interp(n_range_in_degree, in_degree_values, in_degree_distribution)
    total_in_degree_distribution = total_in_degree_distribution/np.sum(total_in_degree_distribution)

    total_out_degree_distribution = np.interp(n_range_out_degree, out_degree_values, out_degree_distribution)
    total_out_degree_distribution = total_out_degree_distribution/np.sum(total_out_degree_distribution)
    return total_in_degree_distribution, total_out_degree_distribution


# generates BARABÁSI AND ALBERT graph from chapter 14.2 "Networks: an introduction" (2010)
# https://math.bme.hu/~gabor/oktatas/SztoM/Newman_Networks.pdf
def ba_graph(sigma_random_variable, a_arbitrary_constant, m_0_nodes, rho_probability, rng, total_nodes):
    er_graph = constant_probability_random_connectivity_matrix(m_0_nodes, rho_probability, rng)
    in_degrees_er_graph = np.sum(er_graph, axis=1)
    list_of_targets = []
    for target, in_degree in enumerate(in_degrees_er_graph):
        for in_degree_counter in range(in_degree):
            list_of_targets.append(target)

    list_of_vertices = [*range(m_0_nodes)]
    total_graph = np.block([
        [er_graph, np.zeros((m_0_nodes, total_nodes - m_0_nodes))],
        [np.zeros((total_nodes - m_0_nodes, m_0_nodes)), np.zeros((total_nodes - m_0_nodes, total_nodes - m_0_nodes))],
    ])
    node_counter = m_0_nodes
    while node_counter < total_nodes:
        for out_degree_counter in range(sigma_random_variable):
            if sigma_random_variable > len(list_of_vertices):
                print("What should happen in this case???")
            r_random_variable = rng.random()
            if r_random_variable < 0.5:
                target_node = rng.choice(list_of_targets)
            else:
                target_node = rng.choice(list_of_vertices)
            list_of_targets.append(target_node)
            total_graph[target_node, node_counter] = 1
        list_of_vertices.append(node_counter)
        node_counter += 1
    return total_graph
# in the matlab files of "Graph-theoretical derivation of brain structural connectivity":
# (https://doi.org/10.1016/j.amc.2020.125150) they call sigma --> gamma, and it is a distribution. They add vertices,
# and if the number of targets (which is a sample taken from the cumulative gamma distribution) is larger than the
# already existing vertices, they take a new sample. Furthermore, gamma is acquired from the original indegree data in
# multiple steps: where one of them consists of interpolating the original data to data which has data for every
# indegree between 0 and the highest observed indegree in the original data. They also allow


def spatial_block_ba_graph(num_cells, x_max, x_min, y_max, y_min, z_max, z_min, delta, noise_factor, m, rng,
                           in_degree_elements, out_degree_elements):
    connectivity_graph = np.zeros((num_cells, num_cells))
    S_N = 200
    S_F = 1
    x_pos_nodes = rng.uniform(x_min, x_max, num_cells)
    y_pos_nodes = rng.uniform(y_min, y_max, num_cells)
    z_pos_nodes = rng.uniform(z_min, z_max, num_cells)
    pos_nodes = np.vstack((x_pos_nodes, y_pos_nodes, z_pos_nodes))
    pos_nodes[:, 0] = np.array([(x_max-x_min)/2, (y_max-y_min)/2, (z_max-z_min)/2])

    centrality_nodes = np.zeros(num_cells)
    for node_index in range(1, num_cells):
        current_pos = pos_nodes[:, node_index]
        dist = np.linalg.norm(current_pos[:, None] - pos_nodes[:, :node_index], axis=0)**2
        random_vector = rng.uniform(size=node_index)
        cost_function = delta*(dist + S_N * noise_factor * random_vector)/S_F + centrality_nodes[:node_index] + 1
        number_of_connections = int(rng.choice(in_degree_elements))
        if number_of_connections >= len(cost_function):
            number_of_connections = len(cost_function)
        if number_of_connections != 0:
            connected_to_nodes = np.argpartition(cost_function, number_of_connections - 1)[:number_of_connections]
            smallest_centrality = np.min(centrality_nodes[connected_to_nodes])
            centrality_nodes[node_index] = smallest_centrality + 1
            connectivity_graph[node_index, connected_to_nodes] = 1

    return connectivity_graph, pos_nodes, centrality_nodes


def convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions, N_total_nodes, p_probability,
                          phi_U_probability, phi_D_probability, rng, sigma_random_variable=3, a_arbitrary_constant=0,
                          delta=1.5, noise_factor=3, in_degree_elements=None, out_degree_elements=None, spatial=False):
    if spatial:
        A_1 = spatial_block_ba_graph(round(N_total_nodes/2), 1, 0, 1, 0, 1, 0, delta, noise_factor, m_0_nodes, rng,
                                     in_degree_elements, out_degree_elements)[0]
        A_2 = spatial_block_ba_graph(round(N_total_nodes/2), 1, 0, 1, 0, 1, 0, delta, noise_factor, m_0_nodes, rng,
                                     in_degree_elements, out_degree_elements)[0]
    else:
        A_1 = ba_graph(sigma_random_variable, a_arbitrary_constant, m_0_nodes, rho_probability, rng, N_total_nodes)
        A_2 = ba_graph(sigma_random_variable, a_arbitrary_constant, m_0_nodes, rho_probability, rng, N_total_nodes)
    B = np.block([
        [A_1, np.zeros(A_1.shape)],
        [np.zeros(A_1.shape), A_2]
    ])
    M_total_partitions = round(N_total_nodes/l_cardinality_partitions)
    nodes_partitions = np.arange(N_total_nodes)
    rng1 = np.random.default_rng(round(rng.random()*100))
    rng2 = np.random.default_rng(round(rng.random()*500))
    A_1_partitions = get_partition(nodes_partitions, l_cardinality_partitions, M_total_partitions, rng1)
    A_2_partitions = get_partition(nodes_partitions, l_cardinality_partitions, M_total_partitions, rng2)
    for i_partition in range(M_total_partitions):
        for j_partition in range(M_total_partitions):
            r_1 = rng.random()
            if r_1 < p_probability:
                s_1 = rng.random()
                if s_1 < phi_U_probability:
                    B[A_2_partitions[j_partition], A_1_partitions[i_partition]] = 1
            else:
                s_1 = rng.random()
                if s_1 < phi_D_probability:
                    B[A_2_partitions[j_partition], A_1_partitions[i_partition]] = 1

            r_2 = rng.random()
            if r_2 < p_probability:
                s_2 = rng.random()
                if s_2 < phi_U_probability:
                    B[A_1_partitions[i_partition], A_2_partitions[j_partition]] = 1
            else:
                s_2 = rng.random()
                if s_2 < phi_D_probability:
                    B[A_1_partitions[i_partition], A_2_partitions[j_partition]] = 1
    return B


# NEEDS TO BE FIXED STILL!!!
# def convolutive_probabilities(in_degree_distribution, out_degree_distribution, N_total_nodes,
#                               l_cardinality_partitions, p_probability, phi_U_probability, phi_D_probability):
#     M = round(N_total_nodes/l_cardinality_partitions)
#     print(M)
#     in_degree_model = []
#     out_degree_model = []
#     delta_0_distribution = np.zeros(N_total_nodes)
#     delta_0_distribution[0] = 1
#     range_to_N = np.arange(N_total_nodes, dtype=float)
#     # NEEDS TO BE FIXED STILL
#     binomial_phi_D_distribution = \
#         (phi_D_probability ** range_to_N) * ((1-phi_D_probability)**(l_cardinality_partitions-range_to_N)) * \
#         comb(l_cardinality_partitions, range_to_N)
#     binomial_phi_U_distribution = \
#         (phi_U_probability ** range_to_N) * ((1 - phi_U_probability) ** (l_cardinality_partitions - range_to_N)) * \
#         comb(l_cardinality_partitions, range_to_N)
#     print(comb(l_cardinality_partitions, range_to_N))
#     print(binomial_phi_U_distribution)
#     print(binomial_phi_D_distribution)
#     for k in range(1, N_total_nodes+1):
#         left_part = (1-p_probability)*delta_0_distribution[:k] + p_probability*binomial_phi_U_distribution[:k]
#         right_part = p_probability*delta_0_distribution[:k] + (1-p_probability)*binomial_phi_D_distribution[:k]
#         first_convolution = np.convolve(left_part, right_part)
#         power_convolutions = convolve_power(first_convolution, M)
#         in_degree_model.append(np.convolve(in_degree_distribution[:k], power_convolutions))
#         out_degree_model.append(np.convolve(out_degree_distribution[:k], power_convolutions))
#     return in_degree_model, out_degree_model


def convolve_power(array, power):
    result = array
    for convolution_iteration in range(power-1):
        result = np.convolve(result, array)
    return result


def get_partition(nodes, l_cardinality, M_partitions, rng):
    rng.shuffle(nodes)
    shuffled_nodes = nodes.copy()
    partitioned_graph = []
    for partition in range(M_partitions):
        partitioned_graph.append(
            shuffled_nodes[partition * l_cardinality:(partition + 1) * l_cardinality])
    return partitioned_graph


def get_csv_degree_elements(in_degree_file_name, out_degree_file_name):
    with open("../degree_data/" + in_degree_file_name) as in_degree_file:
        csv_reader = csv.reader(in_degree_file, delimiter=';')
        in_degree_elements = []
        for row in csv_reader:
            for value in row:
                in_degree_elements.append(int(value))
    with open("../degree_data/" + out_degree_file_name) as out_degree_file:
        csv_reader = csv.reader(out_degree_file, delimiter=';')
        out_degree_elements = []
        for row in csv_reader:
            for value in row:
                out_degree_elements.append(int(value))
    return in_degree_elements, out_degree_elements


def plot_degree_counts(connectivity_graph):
    fig_indegrees, ax_indegrees = plt.subplots()
    indegrees, counts = np.unique(np.sum(connectivity_graph, axis=1), return_counts=True)
    # print(indegrees)
    ax_indegrees.bar(indegrees, counts, align='center')
    ax_indegrees.set_xlabel("indegree")
    ax_indegrees.set_ylabel("count")

    fig_outdegrees, ax_outdegrees = plt.subplots()
    outdegrees, counts = np.unique(np.sum(connectivity_graph, axis=0), return_counts=True)
    ax_outdegrees.bar(outdegrees, counts, align='center')
    ax_outdegrees.set_xlabel("outdegree")
    ax_outdegrees.set_ylabel("count")

