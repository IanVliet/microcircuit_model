import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time


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
    return rng.uniform(size=(num_cells, num_cells)) <= probability


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


def convolutive_graph_gen(sigma_random_variable, a_arbitrary_constant, m_0_nodes, rho_probability,
                          l_cardinality_partitions, N_total_nodes, p_probability, phi_U_probability,
                          phi_D_probability, rng):
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


def get_partition(nodes, l_cardinality, M_partitions, rng):
    rng.shuffle(nodes)
    shuffled_nodes = nodes.copy()
    partitioned_graph = []
    for partition in range(M_partitions):
        partitioned_graph.append(
            shuffled_nodes[partition * l_cardinality:(partition + 1) * l_cardinality])
    return partitioned_graph
