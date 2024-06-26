import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
import csv
from collections import Counter
import networkx as nx
from scipy.special import comb
from scipy.stats import binom
import os
import json
import sys
from microcircuit_model.connectivity.data_preparation.data_prep_utils import *


def get_degree_data(option, data_location='data_preparation'):
    if option == "manc":
        with open(data_location + '/manc_v1.0_indegrees.npy', 'rb') as indegree_file:
            in_degree_elements = np.load(indegree_file)

        with open(data_location + '/manc_v1.0_outdegrees.npy', 'rb') as outdegree_file:
            out_degree_elements = np.load(outdegree_file)
    elif option == "hemibrain":
        with open(data_location + '/hemibrain_v1.2.1_indegrees.npy', 'rb') as indegree_file:
            in_degree_elements = np.load(indegree_file)

        with open(data_location + '/hemibrain_v1.2.1_outdegrees.npy', 'rb') as outdegree_file:
            out_degree_elements = np.load(outdegree_file)
    elif option == "pyc-pyc":
        with open(data_location + '/pyc-pyc_indegrees.npy', 'rb') as indegree_file:
            in_degree_elements = np.load(indegree_file)
        with open(data_location + '/pyc-pyc_outdegrees.npy', 'rb') as outdegree_file:
            out_degree_elements = np.load(outdegree_file)
    # from MICrONS Explorer https://www.microns-explorer.org/phase1, raw data at: https://zenodo.org/records/5579388
    # accompanying paper: Turner, N. L., Macrina, T., Bae, J. A., Yang, R., Wilson, A. M., Schneider-Mizell, C., Lu,
    # R., Wu, J., Bodor, A. L., Bleckert, A. A., Brittain, D., Froudarakis, E., Dorkenwald, S., Collman, F., Kemnitz,
    # N., Ih, D., Silversmith, W. M., Zung, J., … Seung, H. S. (2022). Reconstruction of neocortex: Organelles,
    # compartments, cells, circuits, and activity. Cell, 185(6), 1082-1100.e24. https://doi.org/10.1016/j.cell.2022.01.023
    elif option == "cortical_microns":
        with open(data_location + '/cortical_microns_indegrees.npy', 'rb') as indegree_file:
            in_degree_elements = np.load(indegree_file)
        with open(data_location + '/cortical_microns_outdegrees.npy', 'rb') as outdegree_file:
            out_degree_elements = np.load(outdegree_file)
    #     Also from MICrONS Explorer, however data can be downloaded from
    #     https://www.microns-explorer.org/mm3-release-notes-v117
    else:
        print("The chosen option", option, "is not one of the possible options.")
        exit(1)
    return in_degree_elements, out_degree_elements


def randomly_select_subset_degree_data(in_degree_elements, out_degree_elements, rng, subset=0.5):
    """Randomly selects a subset of the degree elements, e.g. if half (subset=0.5) is selected the indegree elements and
     outdegree elements are randomized and half of the array is saved."""
    num_in_degree_elements_subset = round(len(in_degree_elements)*subset)
    num_out_degree_elements_subset = round(len(out_degree_elements)*subset)

    random_in_degree_elements = rng.permutation(in_degree_elements)
    random_out_degree_elements = rng.permutation(out_degree_elements)

    subset_in_degree_elements = random_in_degree_elements[:num_in_degree_elements_subset]
    subset_out_degree_elements = random_out_degree_elements[:num_out_degree_elements_subset]

    rest_in_degree_elements = random_in_degree_elements[num_in_degree_elements_subset:]
    rest_out_degree_elements = random_out_degree_elements[num_out_degree_elements_subset:]

    return subset_in_degree_elements, subset_out_degree_elements, rest_in_degree_elements, rest_out_degree_elements


def find_small_divisors(number, max_value):
    divisors = []
    for divisor in range(1, number + 1):
        if divisor >= max_value:
            break
        if number % divisor == 0:
            divisors.append(divisor)
    return divisors


def constant_probability_random_connectivity_matrix(num_cells, probability, rng):
    connectivity_matrix = rng.uniform(size=(num_cells, num_cells)) <= probability
    np.fill_diagonal(connectivity_matrix, 0)
    return connectivity_matrix


def get_degree_distributions(in_degree_elements, out_degree_elements):
    in_degree_values, in_degree_counts = np.unique(in_degree_elements, return_counts=True)
    out_degree_values, out_degree_counts = np.unique(out_degree_elements, return_counts=True)
    in_degree_distribution = in_degree_counts / np.sum(in_degree_counts)
    out_degree_distribution = out_degree_counts / np.sum(out_degree_counts)
    return in_degree_values, in_degree_distribution, out_degree_values, out_degree_distribution


def get_interpolated_degree_distributions(in_degree_values, in_degree_distribution, out_degree_values,
                                          out_degree_distribution, last_bin_value_indegree, last_bin_value_outdegree):
    n_range_in_degree = np.arange(last_bin_value_indegree + 1)
    n_range_out_degree = np.arange(last_bin_value_outdegree + 1)
    total_in_degree_distribution = np.interp(n_range_in_degree, in_degree_values, in_degree_distribution)
    # total_in_degree_distribution = total_in_degree_distribution/np.sum(total_in_degree_distribution)

    total_out_degree_distribution = np.interp(n_range_out_degree, out_degree_values, out_degree_distribution)
    # total_out_degree_distribution = total_out_degree_distribution/np.sum(total_out_degree_distribution)
    return total_in_degree_distribution, total_out_degree_distribution


def get_filled_degree_distributions(in_degree_values, in_degree_distribution, out_degree_values,
                                    out_degree_distribution, last_bin_value_indegree, last_bin_value_outdegree):
    total_in_degree_distribution = np.zeros(last_bin_value_indegree + 1)

    for value_in, prob_in in zip(in_degree_values, in_degree_distribution):
        total_in_degree_distribution[value_in] = prob_in

    total_out_degree_distribution = np.zeros(last_bin_value_outdegree + 1)

    for value_out, prob_out in zip(out_degree_values, out_degree_distribution):
        total_out_degree_distribution[value_out] = prob_out

    return total_in_degree_distribution, total_out_degree_distribution


# generates BARABÁSI AND ALBERT (BA) graph based on the code in MATLAB shared in "Graph-theoretical derivation of brain
# structural connectivity": (https://doi.org/10.1016/j.amc.2020.125150)
def ba_graph(sigma, a_arbitrary_constant, m_0_nodes, rho_probability, rng, total_nodes):
    er_graph = constant_probability_random_connectivity_matrix(m_0_nodes, rho_probability, rng)
    out_degrees_er_graph = np.sum(er_graph, axis=0)
    in_degrees_er_graph = np.sum(er_graph, axis=1)
    total_out_degrees = np.zeros(total_nodes)
    total_out_degrees[:m_0_nodes] = out_degrees_er_graph
    total_in_degrees = np.zeros(total_nodes)
    total_in_degrees[:m_0_nodes] = in_degrees_er_graph
    cum_sigma = np.cumsum(sigma)

    # possible_targets = []
    # for target, out_degree in enumerate(out_degrees_er_graph):
    #     for out_degree_counter in range(out_degree + a_arbitrary_constant):
    #         possible_targets.append(target)

    targets = np.arange(m_0_nodes, dtype=int)
    total_graph = np.block([
        [er_graph, np.zeros((m_0_nodes, total_nodes - m_0_nodes))],
        [np.zeros((total_nodes - m_0_nodes, m_0_nodes)), np.zeros((total_nodes - m_0_nodes, total_nodes - m_0_nodes))],
    ])
    node_counter = m_0_nodes
    while node_counter < total_nodes:
        if node_counter % 5000 == 0:
            print("Node counter: " + str(node_counter) + " of " + str(total_nodes))
        num_incoming_connections = len(targets)
        total_out_degrees[targets] += 1
        total_in_degrees[node_counter] += num_incoming_connections
        total_graph[node_counter, targets] = 1
        # the probability for a node to be chosen as a target is related to the outdegree + constant a
        ratio_target = total_out_degrees[:node_counter] + a_arbitrary_constant
        ratio_target = ratio_target/np.sum(ratio_target)
        bins = np.concatenate((np.array([0]), np.cumsum(ratio_target)))
        # choose number of connections from the distribution gamma
        num_connections = node_counter + 1
        while num_connections > node_counter:
            r = rng.random()
            num_connections = np.argmax(r < cum_sigma)
        targets = np.zeros(num_connections, dtype=int) - 1
        connection_counter = 0
        while connection_counter < num_connections:
            random_ratio_target = rng.random()
            target_node = np.digitize(random_ratio_target, bins) - 1
            if target_node not in targets:
                targets[connection_counter] = target_node
                connection_counter += 1
        node_counter += 1
    return total_graph
# in the matlab files of "Graph-theoretical derivation of brain structural connectivity":
# (https://doi.org/10.1016/j.amc.2020.125150) they call sigma --> gamma, and it is a distribution. They add vertices,
# and if the number of targets (which is a sample taken from the cumulative gamma distribution) is larger than the
# already existing vertices, they take a new sample. Furthermore, gamma is acquired from the original indegree data in
# multiple steps: where one of them consists of interpolating the original data to data which has data for every
# indegree between 0 and the highest observed indegree in the original data.


def spatial_block_ba_graph(num_cells, x_max, x_min, y_max, y_min, z_max, z_min, delta, noise_factor, m_0_nodes, rng,
                           sigma, rho_probability):
    connectivity_graph = np.zeros((num_cells, num_cells), dtype=bool)
    connectivity_graph[:m_0_nodes, :m_0_nodes] = random_graph = constant_probability_random_connectivity_matrix(m_0_nodes,
                                                                                                 rho_probability, rng)

    S_N = 200
    S_F = 1
    x_pos_nodes = rng.uniform(x_min, x_max, num_cells)
    y_pos_nodes = rng.uniform(y_min, y_max, num_cells)
    z_pos_nodes = rng.uniform(z_min, z_max, num_cells)
    pos_nodes = np.vstack((x_pos_nodes, y_pos_nodes, z_pos_nodes))
    pos_nodes[:, 0] = np.array([(x_max-x_min)/2, (y_max-y_min)/2, (z_max-z_min)/2])

    cum_sigma = np.cumsum(sigma)
    # print(cum_sigma)
    centrality_nodes = np.zeros(num_cells)
    directed_graph = nx.DiGraph(random_graph)
    shortest_paths = nx.single_source_dijkstra_path_length(directed_graph, 0)
    for node, distance in shortest_paths.items():
        centrality_nodes[node] = distance
    # SHOULD RECHECK INDICES OF NODES!!!
    for node_index in range(m_0_nodes + 1, num_cells):
        current_pos = pos_nodes[:, node_index]
        dist = np.linalg.norm(current_pos[:, None] - pos_nodes[:, :node_index], axis=0)**2
        random_vector = rng.uniform(size=node_index)
        cost_function = delta*(dist + S_N * noise_factor * random_vector)/S_F + centrality_nodes[:node_index] + 1
        number_of_connections = node_index + 1
        attempts = 0
        while number_of_connections > node_index:
            r = rng.random()
            number_of_connections = np.argmax(r < cum_sigma)
            attempts += 1
            if attempts > 10000:
                print("looping for too long, seems unable to get the number of connections smaller than: "+str(node_index))
                number_of_connections = node_index
        if number_of_connections != 0:
            connected_to_nodes = np.argpartition(cost_function, number_of_connections - 1)[:number_of_connections]
            smallest_centrality = np.min(centrality_nodes[connected_to_nodes])
            centrality_nodes[node_index] = smallest_centrality + 1
            connectivity_graph[node_index, connected_to_nodes] = 1

    return connectivity_graph, pos_nodes, centrality_nodes


def convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions, N_total_nodes, E_K,
                          phi_U_probability, phi_D_probability, rng, in_degree_elements, out_degree_elements,
                          dimensions, pc=300, delta=3, noise_factor=1.5, spatial=True, return_positions=False):

    N_1 = N_2 = round(N_total_nodes/2)
    p_probability = (E_K / N_1 - phi_D_probability) / (phi_U_probability - phi_D_probability)
    print("p: " + str(p_probability))
    m = np.mean(in_degree_elements)
    print("m: " + str(m))
    # in_degree_values, in_degree_distribution, out_degree_values, out_degree_distribution = \
    #     get_degree_distributions(in_degree_elements, out_degree_elements)
    # in_degree, out_degree = \
    #     get_interpolated_degree_distributions(in_degree_values, in_degree_distribution, out_degree_values,
    #                                           out_degree_distribution)
    # print("indegree: " + str(in_degree))
    max_indegree = int(np.max(in_degree_elements))
    max_outdegree = int(np.max(out_degree_elements))
    bins_indegree_hist = np.arange(0, max_indegree + 1, 5)
    pre_in_degree_distribution, pre_bins_indegree = np.histogram(in_degree_elements, bins=bins_indegree_hist, density=True)
    in_degree_distribution = np.append([0], pre_in_degree_distribution[pre_in_degree_distribution > 0])
    in_degree_values = np.append([0], (pre_bins_indegree[:-1][pre_in_degree_distribution > 0] + pre_bins_indegree[1:][pre_in_degree_distribution > 0])/2)

    bins_outdegree_hist = np.arange(0, max_outdegree + 1, 5)
    pre_out_degree_distribution, pre_bins_outdegree = np.histogram(out_degree_elements, bins=bins_outdegree_hist, density=True)
    out_degree_distribution = np.append([0], pre_out_degree_distribution[pre_out_degree_distribution > 0])
    out_degree_values = np.append([0], (pre_bins_outdegree[:-1][pre_out_degree_distribution > 0] + pre_bins_outdegree[1:][pre_out_degree_distribution > 0]) / 2)

    last_bin_value_indegree = int(np.floor(in_degree_values[-1]))
    last_bin_value_outdegree = int(np.floor(out_degree_values[-1]))
    in_degree, out_degree = \
        get_interpolated_degree_distributions(in_degree_values, in_degree_distribution, out_degree_values,
                                              out_degree_distribution, last_bin_value_indegree, last_bin_value_outdegree)

    ni = np.arange(1, last_bin_value_indegree + 1)
    weighted_indegree = np.arange(last_bin_value_indegree + 1)*in_degree
    indegree_mean = np.sum(weighted_indegree)
    # print(len(in_degree))
    # print(indegree_mean)
    cumsum_weighted_indegree = np.cumsum(weighted_indegree)
    # print("cumsum_weighted_indegree: " + str(cumsum_weighted_indegree))
    sum_indegree = np.sum(in_degree)
    # print("sum_indegree: " + str(sum_indegree))
    cumsum_indegree = np.cumsum(in_degree)
    # print("cumsum_indegree: " + str(cumsum_indegree))
    var_phi = \
        ((indegree_mean - cumsum_weighted_indegree) - np.arange(1, last_bin_value_indegree + 2)*(sum_indegree - cumsum_indegree)) \
        / (sum_indegree - cumsum_indegree)
    # print("var_phi: " + str(var_phi))
    # fig, ax = plt.subplots()
    # ax.plot((indegree_mean - cumsum_weighted_indegree), label="weighted in degree")
    # ax.plot(np.arange(1, last_bin_value_indegree + 2)*(sum_indegree - cumsum_indegree)/(sum_indegree - cumsum_indegree),
    #         label="weights")
    # ax.plot((sum_indegree - cumsum_indegree), label="division")
    # ax.plot(var_phi, label=r"$\varphi$")
    # ax.axhline(m - E_K, color='red')
    # ax.legend()
    d_index = np.argmax(var_phi < (m - E_K))
    print("d_index: " + str(d_index))
    alpha_k = np.zeros(ni[-1])
    # print("in_degree: " + str(in_degree))
    alpha_k[:ni[-1] - d_index] = in_degree[d_index+1:]
    alpha_k = alpha_k/np.sum(alpha_k)
    # print("alpha_k: " + str(alpha_k))
    binomial_rho_k = binom.pmf(np.arange(last_bin_value_indegree), m_0_nodes-1, rho_probability)
    # print(sum(binomial_rho_k))
    sigma = N_1/(N_1 - m_0_nodes)*alpha_k - m_0_nodes/(N_1 - m_0_nodes)*binomial_rho_k
    sigma = sigma/np.sum(sigma)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(alpha_k, label=r"$\alpha_k$")
    # ax2.plot(binomial_rho_k, label="binomial")
    # ax2.plot(sigma, label=r"$\sigma$")
    # ax2.legend()
    # plt.show()
    # print("sigma: " + str(sigma))
    positions_A_1 = np.vstack((np.zeros(N_1), np.zeros(N_1), np.zeros(N_1)))
    positions_A_2 = np.vstack((np.zeros(N_2), np.zeros(N_2), np.zeros(N_2)))
    if spatial:
        A_1, positions_A_1 = spatial_block_ba_graph(N_1, dimensions[0], dimensions[1], dimensions[2], dimensions[3],
                                     dimensions[4], dimensions[5], delta, noise_factor, m_0_nodes, rng,
                                     sigma, rho_probability)[0:2]
        A_2, positions_A_2 = spatial_block_ba_graph(N_2, dimensions[0], dimensions[1], dimensions[2], dimensions[3],
                                     dimensions[4], dimensions[5], delta, noise_factor, m_0_nodes, rng,
                                     sigma, rho_probability)[0:2]
    else:
        c = (m - E_K - m_0_nodes/N_1 * (m_0_nodes - 1) * rho_probability) * N_1 / (N_1 - m_0_nodes)
        print("c: " + str(c))
        a = round(pc*c)
        print("a: " + str(a))
        if a < 0:
            print("a is uncharacteristically small")
        start_A_1 = time.time()
        A_1 = ba_graph(sigma, a, m_0_nodes, rho_probability, rng, N_1)
        end_A_1 = time.time()

        in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix=A_1)
        fig_ba, ax_ba = plt.subplots(2)
        num_bins = 100
        ax_ba[0].hist(in_degree_elements_model, label='sim', alpha=0.8, bins=num_bins, density=True, edgecolor='black',
                   linewidth=1)
        ax_ba[0].set_xlabel("indegree")
        ax_ba[0].set_ylabel("probability")
        ax_ba[1].hist(out_degree_elements_model, label='sim', alpha=0.8, bins=num_bins, density=True, edgecolor='black',
                   linewidth=1)
        ax_ba[1].set_xlabel("outdegree")
        ax_ba[1].set_ylabel("probability")
        ax_ba[0].legend()
        ax_ba[1].legend()

        A_2 = ba_graph(sigma, a, m_0_nodes, rho_probability, rng, N_2)
        end_A_2 = time.time()
        print("A_1 generation: " + str(end_A_1 - start_A_1))
        print("A_2 generation: " + str(end_A_2 - end_A_1))
    B = np.block([
        [A_1.astype(bool), np.zeros(A_1.shape, dtype=bool)],
        [np.zeros(A_1.shape, dtype=bool), A_2.astype(bool)]
    ])
    M_total_partitions = round(N_1/l_cardinality_partitions)
    nodes_partitions_A_1 = np.arange(N_1)
    nodes_partitions_A_2 = np.arange(N_1, N_total_nodes)
    rng1 = np.random.default_rng(round(rng.random()*100))
    rng2 = np.random.default_rng(round(rng.random()*500))
    start_partition_generation = time.time()
    A_1_partitions = get_partition(nodes_partitions_A_1, l_cardinality_partitions, M_total_partitions, rng1)
    A_2_partitions = get_partition(nodes_partitions_A_2, l_cardinality_partitions, M_total_partitions, rng2)
    # print(A_1_partitions)
    end_partition_generation = time.time()
    print("Generate partitions: " + str(end_partition_generation - start_partition_generation))
    for i_partition in range(M_total_partitions):
        for j_partition in range(M_total_partitions):
            r_1 = rng.random()
            if r_1 < p_probability:
                B[A_2_partitions[j_partition], A_1_partitions[i_partition].T] = \
                    rng.random((l_cardinality_partitions, l_cardinality_partitions)) < phi_U_probability
                # s_1 = rng.random()
                # if s_1 < phi_U_probability:
                #     # print("before: " + str(B[A_2_partitions[j_partition], A_1_partitions[i_partition]]))
                #     B[A_2_partitions[j_partition], A_1_partitions[i_partition].T] = 1
                #     # print("after: " + str(B[A_2_partitions[j_partition], A_1_partitions[i_partition]]))
            else:
                B[A_2_partitions[j_partition], A_1_partitions[i_partition].T] = \
                    rng.random((l_cardinality_partitions, l_cardinality_partitions)) < phi_D_probability
                # s_1 = rng.random()
                # if s_1 < phi_D_probability:
                #     B[A_2_partitions[j_partition], A_1_partitions[i_partition].T] = 1

            r_2 = rng.random()
            if r_2 < p_probability:
                B[A_1_partitions[i_partition], A_2_partitions[j_partition].T] = \
                    rng.random((l_cardinality_partitions, l_cardinality_partitions)) < phi_U_probability
                # s_2 = rng.random()
                # if s_2 < phi_U_probability:
                #     B[A_1_partitions[i_partition], A_2_partitions[j_partition].T] = 1
            else:
                B[A_1_partitions[i_partition], A_2_partitions[j_partition].T] = \
                    rng.random((l_cardinality_partitions, l_cardinality_partitions)) < phi_D_probability
                # s_2 = rng.random()
                # if s_2 < phi_D_probability:
                #     B[A_1_partitions[i_partition], A_2_partitions[j_partition].T] = 1
    end_exponential_model = time.time()
    print("Exponential model generation: " + str(end_exponential_model - end_partition_generation))
    if return_positions:
        return B, np.concatenate((positions_A_1, positions_A_2), axis=1)
    else:
        return B


def simple_ba_graph(total_nodes, m_0_nodes, m, rng, a_arbitrary_constant=1000):
    if m > m_0_nodes:
        exit(1)
    er_graph = constant_probability_random_connectivity_matrix(m_0_nodes, 1, rng)
    out_degrees_er_graph = np.sum(er_graph, axis=0)
    in_degrees_er_graph = np.sum(er_graph, axis=1)
    total_out_degrees = np.zeros(total_nodes)
    total_out_degrees[:m_0_nodes] = out_degrees_er_graph
    total_in_degrees = np.zeros(total_nodes)
    total_in_degrees[:m_0_nodes] = in_degrees_er_graph

    # possible_targets = []
    # for target, out_degree in enumerate(out_degrees_er_graph):
    #     for out_degree_counter in range(out_degree + a_arbitrary_constant):
    #         possible_targets.append(target)

    targets = np.arange(m_0_nodes, dtype=int)
    total_graph = np.block([
        [er_graph, np.zeros((m_0_nodes, total_nodes - m_0_nodes))],
        [np.zeros((total_nodes - m_0_nodes, m_0_nodes)), np.zeros((total_nodes - m_0_nodes, total_nodes - m_0_nodes))],
    ])
    num_connections = m
    node_counter = m_0_nodes
    while node_counter < total_nodes:
        if node_counter % 5000 == 0:
            print("Node counter: " + str(node_counter) + " of " + str(total_nodes))
        num_incoming_connections = len(targets)
        total_out_degrees[targets] += 1
        total_in_degrees[node_counter] += num_incoming_connections
        total_graph[node_counter, targets] = 1
        # the probability for a node to be chosen as a target is related to the outdegree + constant a
        ratio_target = total_out_degrees[:node_counter] + a_arbitrary_constant
        ratio_target = ratio_target / np.sum(ratio_target)
        bins = np.concatenate((np.array([0]), np.cumsum(ratio_target)))

        targets = np.zeros(num_connections, dtype=int) - 1
        connection_counter = 0
        while connection_counter < num_connections:
            random_ratio_target = rng.random()
            target_node = np.digitize(random_ratio_target, bins) - 1
            if target_node not in targets:
                targets[connection_counter] = target_node
                connection_counter += 1
        node_counter += 1
    return total_graph


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
        partition_nodes = shuffled_nodes[partition * l_cardinality:(partition + 1) * l_cardinality]
        partition_tile = np.tile(partition_nodes, (l_cardinality, 1))
        partitioned_graph.append(partition_tile)
    return partitioned_graph


def custom_trial_name_creator(trial):
    return f"conv_model{trial.trial_id}"


def custom_file_name_creator(trial):
    return f"conv_model{trial.trial_id}"


def cross_entropy(data_distribution, model_distribution):
    return -np.sum(data_distribution*np.log2(model_distribution[:len(data_distribution)]))


def linear_combination_cross_entropy(in_degree_cross_entropy, out_degree_cross_entropy, weight):
    return weight * in_degree_cross_entropy + (1 - weight) * out_degree_cross_entropy


def degree_elements_are_np_arrays(in_degree_elements, out_degree_elements):
    return isinstance(in_degree_elements, np.ndarray) and isinstance(out_degree_elements, np.ndarray)


def degree_elements_are_lists_of_np_arrays(in_degree_elements, out_degree_elements):
    return (isinstance(in_degree_elements, list) and isinstance(out_degree_elements, list) and
            all(isinstance(in_degree, np.ndarray) for in_degree in in_degree_elements) and
            all(isinstance(out_degree, np.ndarray) for out_degree in out_degree_elements))


def extend_with_zeros(total_length, original_distribution):
    new_distribution = np.zeros(total_length)
    new_distribution[:len(original_distribution)] = original_distribution
    return new_distribution


def produce_connectivity_calculate_cross_entropy(fixed_hyperparameters, in_degree_elements, out_degree_elements, config):

    number_of_seeds = fixed_hyperparameters["number_of_seeds"]
    N_total_nodes = fixed_hyperparameters["N_total_nodes"]
    int_random_generator = fixed_hyperparameters["int_random_generator"]
    weight = fixed_hyperparameters["weight"]
    dimensions = fixed_hyperparameters["dimensions"]
    pc = fixed_hyperparameters["pc"]

    rng = np.random.default_rng(int_random_generator)
    m_0_nodes = config["m_0"]
    rho_probability = config["rho"]  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions = config["l"]  # Likely L in the web application
    E_k = config["E_k"]  # likely EK in the web application
    phi_U_probability = config["phi_U"]  # likely phi_U in the web application
    phi_D_probability = config["phi_D"]  # likely phi_D in the web application
    delta = config["delta"]
    noise_factor = config["noise_factor"]

    child_rgns = rng.spawn(number_of_seeds)
    total_cross_entropy = 0
    for seed_run, child_rng in zip(range(number_of_seeds), child_rgns):
        B = convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions,
                                  N_total_nodes, E_k, phi_U_probability,
                                  phi_D_probability, child_rng, in_degree_elements, out_degree_elements, dimensions,
                                  delta=delta, noise_factor=noise_factor, spatial=True, pc=pc)

        # degree distributions of model
        in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix=B)

        total_cross_entropy += elements_linear_cross_entropy(in_degree_elements_model, out_degree_elements_model,
                                                            in_degree_elements, out_degree_elements, weight)
    average_total_cross_entropy = total_cross_entropy/number_of_seeds
    return {"score": average_total_cross_entropy}


def elements_linear_cross_entropy(in_degree_elements_model, out_degree_elements_model, in_degree_elements, out_degree_elements, weight):
    in_degree_values, in_degree_distribution, out_degree_values, out_degree_distribution = \
        get_degree_distributions(in_degree_elements, out_degree_elements)

    in_degree, out_degree = \
        get_filled_degree_distributions(in_degree_values, in_degree_distribution, out_degree_values,
                                        out_degree_distribution, np.max(in_degree_elements),
                                        np.max(out_degree_elements))

    in_degree_values_model, in_degree_distribution_model, out_degree_values_model, out_degree_distribution_model = \
        get_degree_distributions(in_degree_elements_model, out_degree_elements_model)

    in_degree_gen_model, out_degree_gen_model = \
        get_filled_degree_distributions(in_degree_values_model, in_degree_distribution_model,
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

    return linear_combination_cross_entropy(in_degree_cross_entropy, out_degree_cross_entropy, weight=weight)


def create_data_directory(str_identifier, results_folder="/results_data/", subdirectory=None):
    if not os.path.exists(str_identifier):
        os.makedirs(str_identifier)

    if subdirectory is None:
        subdirectory_name = ""
    else:
        subdirectory_name = subdirectory

    folder_identifier = 0
    while os.path.exists(str_identifier + results_folder + str(folder_identifier) + subdirectory_name):
        folder_identifier += 1
    data_directory = str_identifier + results_folder + str(folder_identifier) + subdirectory_name
    os.makedirs(data_directory)
    return data_directory


def save_connectivity_graphs(data_directory, graph, graph_number):
    with open(data_directory + "/connectivity_graph_" + str(graph_number) + ".npy", "wb") as connectivityfile:
        np.save(connectivityfile, graph)


def get_names_for_hist_and_cross_vs_total_nodes(log_type, N_total_nodes, binsize=None):
    if log_type:
        graph_type_name = "_log"
    else:
        graph_type_name = "_lin"
    if binsize is not None:
        binsize_name = "_binsize_" + str(binsize)
    else:
        binsize_name = ""
    hist_degree_distributions_name = "/hist_degree_distributions" + graph_type_name + "_N_" + str(N_total_nodes) + binsize_name
    cross_entropy_network_size_name = "/cross_entropy_network_size" + graph_type_name + binsize_name
    return hist_degree_distributions_name, cross_entropy_network_size_name


def get_fixed_hyperparameters(str_identifier, fixed_hyperparameters_name, int_random_generator=None,
                              N_total_nodes=None, weight=None, number_of_seeds=None, dimensions=None):
    try:
        with open(str_identifier + fixed_hyperparameters_name, "r") as configfile:
            json_parameters = json.load(configfile)
        if int_random_generator is None:
            int_random_generator = json_parameters['int_random_generator']
        if N_total_nodes is None:
            N_total_nodes = json_parameters['N_total_nodes']
        if weight is None:
            weight = json_parameters['weight']
        option = json_parameters['option']
        if number_of_seeds is None:
            number_of_seeds = json_parameters['number_of_seeds']

        if "dimensions" in json_parameters:
            if dimensions is None:
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
    return json_parameters, fixed_hyperparameters, dim_pc_fixed_parameter


def get_standard_file_and_folder_names(option, str_identifier, fixed_hyperparameters_name, log_type, N_total_nodes):
    save_optimised_parameters = "optimised_parameters/" + option

    file_name = str_identifier + "/top3_results.pkl"
    png_extension = ".png"
    pdf_extension = ".pdf"
    hist_degree_distributions_name, cross_entropy_network_size_name = (
        get_names_for_hist_and_cross_vs_total_nodes(log_type, N_total_nodes))
    detailed_name = "_detailed"

    str_name_dict = {
        "str_identifier": str_identifier,
        "hist_degree_distributions_name": hist_degree_distributions_name,
        "cross_entropy_network_size_name": cross_entropy_network_size_name,
        "png_extension": png_extension,
        "pdf_extension": pdf_extension,
        "detailed_name": detailed_name,
        "fixed_hyperparameters_name": fixed_hyperparameters_name
    }
    return save_optimised_parameters, file_name, str_name_dict


def reproduce_top3_results(top3_results, in_degree_elements, out_degree_elements, str_name_dict, fixed_hyperparameters,
                           dim_pc_fixed_parameter, save_connectivity=True, one_seed_only=False, log_type=True,
                           binsize=20, single_figure=False, top_result_only=False, main_directory="/results_data/",
                           subdirectory=None):
    str_identifier = str_name_dict["str_identifier"]
    fixed_hyperparameters_name = str_name_dict["fixed_hyperparameters_name"]
    # get original hyperparameters
    try:
        with open(str_identifier + fixed_hyperparameters_name, "r") as configfile:
            json_parameters = json.load(configfile)
    except IOError:
        print("Could not process the file due to IOError")
        sys.exit(1)
    except KeyError:
        print("A key was not defined")
        sys.exit(2)

    int_random_generator = fixed_hyperparameters['int_random_generator']
    N_total_nodes = fixed_hyperparameters['N_total_nodes']
    weight = fixed_hyperparameters['weight']
    option = fixed_hyperparameters['option']
    if "number_of_seeds" in fixed_hyperparameters and not one_seed_only:
        number_of_seeds = fixed_hyperparameters['number_of_seeds']
    else:
        number_of_seeds = 1
    dimensions = fixed_hyperparameters['dimensions']
    pc = fixed_hyperparameters['pc']
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
        if not dim_pc_fixed_parameter:
            if "config/dimensions" in result:
                dimensions = result["config/dimensions"]
                pc = result["config/pc"]
            else:
                print("dimensions and pc could not be found in either fixed_hyperparameters.json nor top3_results.pkl")
                exit(-1)

        data_directory = create_data_directory(str_identifier, results_folder=main_directory, subdirectory=subdirectory)
        if number_of_seeds == 1:
            graph = convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions,
                                          N_total_nodes, E_k, phi_U_probability,
                                          phi_D_probability, rng, in_degree_elements, out_degree_elements, dimensions,
                                          delta=delta, noise_factor=noise_factor, spatial=True, pc=pc)
            if save_connectivity:
                save_connectivity_graphs(data_directory, graph, 0)

            # results from generative convolutional model
            in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix=graph)

            plot_and_save_histograms(option, data_directory, str_name_dict, in_degree_elements, out_degree_elements,
                                     in_degree_elements_model, out_degree_elements_model, log_type=log_type, binsize=binsize)

            recreated_score = elements_linear_cross_entropy(in_degree_elements_model, out_degree_elements_model,
                                                            in_degree_elements,
                                                            out_degree_elements, weight)
            std_recreated_score = None
        else:
            recreated_scores = np.zeros(number_of_seeds)
            plot_figure = True
            for seed_number, child_rng in zip(range(number_of_seeds), rng.spawn(number_of_seeds)):
                graph = convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions,
                                              N_total_nodes, E_k, phi_U_probability,
                                              phi_D_probability, child_rng, in_degree_elements, out_degree_elements, dimensions,
                                              delta=delta, noise_factor=noise_factor, spatial=True, pc=pc)
                # save graph
                if save_connectivity:
                    save_connectivity_graphs(data_directory, graph, seed_number)

                # results from generative convolutional model
                in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix=graph)
                if plot_figure:
                    plot_and_save_histograms(option, data_directory, str_name_dict, in_degree_elements, out_degree_elements,
                                             in_degree_elements_model, out_degree_elements_model, log_type=log_type, binsize=binsize)
                    if single_figure:
                        plot_figure = False

                recreated_scores[seed_number] = elements_linear_cross_entropy(in_degree_elements_model,
                                                                              out_degree_elements_model,
                                                                              in_degree_elements,
                                                                              out_degree_elements, weight)
            recreated_score = np.average(recreated_scores)
            std_recreated_score = np.std(recreated_scores, ddof=1)

        print("original score:", result["score"])
        print("recreated score:", recreated_score)
        print("std recreated_score:", std_recreated_score)
        scores = {
            "original score": result["score"],
            "recreated score": recreated_score,
            "std_recreated_score": std_recreated_score,
        }

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

        # Also save hyperparameters that were chosen differently
        for key, value in fixed_hyperparameters.items():
            if json_parameters[key] != value:
                config[key] = value
        print(config)

        with open(data_directory + "/scores_parameters.json", "w") as scores_file:
            json.dump(scores | config, scores_file)

        if top_result_only:
            return


def full_analysis_optimised_connectivity(top3_results, in_degree_elements, out_degree_elements, str_name_dict, fixed_hyperparameters,
                           dim_pc_fixed_parameter, save_connectivity=True, total_nodes_binsize_plot_type_dict=None):

    if total_nodes_binsize_plot_type_dict is None:
        print("The configurations which should be analysed were not specified (correctly)")
        return

    str_identifier = str_name_dict["str_identifier"]
    fixed_hyperparameters_name = str_name_dict["fixed_hyperparameters_name"]

    # get fixed_parameters (could have been adapted)
    int_random_generator = fixed_hyperparameters['int_random_generator']
    weight = fixed_hyperparameters['weight']
    option = fixed_hyperparameters['option']
    if "number_of_seeds" in fixed_hyperparameters:
        number_of_seeds = fixed_hyperparameters['number_of_seeds']
    else:
        number_of_seeds = 5
    dimensions = fixed_hyperparameters['dimensions']
    pc = fixed_hyperparameters['pc']

    # select the result we analyse (standard is that we choose the best result)
    result = top3_results.iloc[0]
    # get parameters specific to result
    rng = np.random.default_rng(int_random_generator)
    m_0_nodes = result["config/m_0"]
    rho_probability = result["config/rho"]  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions = result["config/l"]  # Likely L in the web application
    E_k = result["config/E_k"]  # likely EK in the web application
    phi_U_probability = result["config/phi_U"]  # likely phi_U in the web application
    phi_D_probability = result["config/phi_D"]  # likely phi_D in the web application
    delta = result["config/delta"]
    noise_factor = result["config/noise_factor"]
    if not dim_pc_fixed_parameter:
        if "config/dimensions" in result:
            dimensions = result["config/dimensions"]
            pc = result["config/pc"]
        else:
            print(
                "dimensions and pc could not be found in either fixed_hyperparameters.json nor top3_results.pkl")
            exit(-1)

    # get original hyperparameters
    try:
        with open(str_identifier + fixed_hyperparameters_name, "r") as configfile:
            json_parameters = json.load(configfile)
    except IOError:
        print("Could not process the file due to IOError")
        sys.exit(1)
    except KeyError:
        print("A key was not defined")
        sys.exit(2)
    for N_total_nodes, plot_configurations in total_nodes_binsize_plot_type_dict.items():
        subdirectory = "/total_nodes_N_" + str(N_total_nodes) + "/"
        data_directory = create_data_directory(str_identifier, results_folder="/full_analysis/",
                                               subdirectory=subdirectory)
        recreated_scores = np.zeros(number_of_seeds)
        # We only create a figure and save a connectivity for the first seed,
        # the other seeds are used to recreate the cross-entropy,
        # and calculate a standard deviation of the cross-entropy.
        for seed_number, child_rng in zip(range(number_of_seeds), rng.spawn(number_of_seeds)):
            graph = convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions,
                                          N_total_nodes, E_k, phi_U_probability,
                                          phi_D_probability, child_rng, in_degree_elements, out_degree_elements,
                                          dimensions,
                                          delta=delta, noise_factor=noise_factor, spatial=True, pc=pc)
            # save graph
            if save_connectivity and seed_number == 0:
                save_connectivity_graphs(data_directory, graph, N_total_nodes)

            # results from generative convolutional model
            in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix=graph)

            recreated_scores[seed_number] = elements_linear_cross_entropy(in_degree_elements_model,
                                                                          out_degree_elements_model,
                                                                          in_degree_elements,
                                                                          out_degree_elements, weight)
            # plot the data for the different configurations for the first seed.
            if seed_number == 0:
                for plot_configuration in plot_configurations:
                    binsize, plot_type = plot_configuration
                    if plot_type == 'log':
                        log_type = True
                    elif plot_type == 'lin':
                        log_type = False
                    else:
                        print("The plot_type:", plot_type, "is not one of the options ('log' or 'lin'); skipping")
                        break
                    hist_degree_distributions_name = (
                        get_names_for_hist_and_cross_vs_total_nodes(log_type, N_total_nodes, binsize=binsize))[0]
                    str_name_dict["hist_degree_distributions_name"] = hist_degree_distributions_name
                    plot_and_save_histograms(option, data_directory, str_name_dict, in_degree_elements,
                                             out_degree_elements,
                                             in_degree_elements_model, out_degree_elements_model, log_type=log_type,
                                             binsize=binsize, plot_detailed=False)

        recreated_score = np.average(recreated_scores)
        std_recreated_score = np.std(recreated_scores, ddof=1)

        print("original score:", result["score"])
        print("recreated score:", recreated_score)
        print("std recreated_score:", std_recreated_score)
        scores = {
            "original score": result["score"],
            "recreated score": recreated_score,
            "score_of_this_seed": recreated_scores[0],
            "std_recreated_score": std_recreated_score,
        }

        config = {
            "m_0_nodes": int(m_0_nodes),
            "rho_probability": rho_probability,
            "l_cardinality_partitions": int(l_cardinality_partitions),
            "E_k": E_k,
            "phi_U_probability": phi_U_probability,
            "phi_D_probability": phi_D_probability,
            "delta": delta,
            "noise_factor": noise_factor,
            "dimensions": dimensions,
            "pc": int(pc),
        }

        # Also save hyperparameters that were chosen differently
        for key, value in fixed_hyperparameters.items():
            original_value = json_parameters[key]
            if key in {"m_0_nodes", "l_cardinality_partitions", "pc"}:
                original_value = int(original_value)
            if original_value != value:
                config[key] = value
        print(config)

        with open(data_directory + "/scores_parameters.json", "w") as scores_file:
            json.dump(scores | config, scores_file)


def save_parameters_top_result(data_directory, top3_results, fixed_hyperparameters, dim_pc_fixed_parameter):
    if "number_of_seeds" not in fixed_hyperparameters:
        number_of_seeds = 1
        fixed_hyperparameters['number_of_seeds'] = number_of_seeds

    dimensions = fixed_hyperparameters['dimensions']
    pc = fixed_hyperparameters['pc']

    top_result = top3_results.iloc[0]
    m_0_nodes = top_result["config/m_0"]
    rho_probability = top_result["config/rho"]  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions = top_result["config/l"]  # Likely L in the web application
    E_k = top_result["config/E_k"]  # likely EK in the web application
    phi_U_probability = top_result["config/phi_U"]  # likely phi_U in the web application
    phi_D_probability = top_result["config/phi_D"]  # likely phi_D in the web application
    delta = top_result["config/delta"]
    noise_factor = top_result["config/noise_factor"]
    if not dim_pc_fixed_parameter:
        if "config/dimensions" in top_result:
            dimensions = top_result["config/dimensions"]
            pc = top_result["config/pc"]
        else:
            print("dimensions and pc could not be found in either fixed_hyperparameters.json nor top3_results.pkl")
            exit(-1)
    optimised_parameters = {
            "m_0_nodes": int(m_0_nodes),
            "rho_probability": rho_probability,
            "l_cardinality_partitions": int(l_cardinality_partitions),
            "E_k": E_k,
            "phi_U_probability": phi_U_probability,
            "phi_D_probability": phi_D_probability,
            "delta": delta,
            "noise_factor": noise_factor,
            "dimensions": dimensions,
            "pc": int(pc),
        }

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    with open(data_directory + "/all_optimised_parameters.json", "w") as parameters_file:
        json.dump(fixed_hyperparameters | optimised_parameters, parameters_file)
        
        
def plot_cross_entropy_vs_network_size(result, in_degree_elements, out_degree_elements, fixed_hyperparameters,
                                       str_name_dict, dim_pc_fixed_parameter, network_sizes,
                                       save_connectivity=False, log_type=False, fig_and_ax=None, save_plot=True):
    print("Creating connectivity's using network_sizes.")
    str_identifier = str_name_dict["str_identifier"]

    optimised_N_total_nodes = fixed_hyperparameters["N_total_nodes"]
    int_random_generator = fixed_hyperparameters['int_random_generator']
    weight = fixed_hyperparameters['weight']
    option = fixed_hyperparameters['option']
    if "number_of_seeds" in fixed_hyperparameters:
        number_of_seeds = fixed_hyperparameters['number_of_seeds']
    else:
        number_of_seeds = 5
    print("Number of seeds used: ", number_of_seeds)

    dimensions = fixed_hyperparameters['dimensions']
    pc = fixed_hyperparameters['pc']

    rng = np.random.default_rng(int_random_generator)
    m_0_nodes = result["config/m_0"]
    rho_probability = result["config/rho"]  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions = result["config/l"]  # Likely L in the web application
    E_k = result["config/E_k"]  # likely EK in the web application
    phi_U_probability = result["config/phi_U"]  # likely phi_U in the web application
    phi_D_probability = result["config/phi_D"]  # likely phi_D in the web application
    delta = result["config/delta"]
    noise_factor = result["config/noise_factor"]
    if not dim_pc_fixed_parameter:
        if "config/dimensions" in result:
            dimensions = result["config/dimensions"]
            pc = result["config/pc"]
        else:
            print(
                "dimensions and pc could not be found in either fixed_hyperparameters.json nor top3_results.pkl")
            exit(-1)

    data_directory = create_data_directory(str_identifier, "/cross_entropy_network_sizes/")
    network_sizes_avg_scores = np.zeros(len(network_sizes))
    network_sizes_std_scores = np.zeros(len(network_sizes))
    for size_index, N_total_nodes in enumerate(network_sizes):
        recreated_scores = np.zeros(number_of_seeds)
        for seed_number, child_rng in zip(range(number_of_seeds), rng.spawn(number_of_seeds)):
            graph = convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions,
                                          N_total_nodes, E_k, phi_U_probability,
                                          phi_D_probability, child_rng, in_degree_elements, out_degree_elements,
                                          dimensions,
                                          delta=delta, noise_factor=noise_factor, spatial=True, pc=pc)
            # save graph
            if save_connectivity:
                save_connectivity_graphs(data_directory, graph, seed_number)

            # results from generative convolutional model
            in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix=graph)

            recreated_scores[seed_number] = elements_linear_cross_entropy(in_degree_elements_model,
                                                                          out_degree_elements_model,
                                                                          in_degree_elements,
                                                                          out_degree_elements, weight)
        network_sizes_avg_scores[size_index] = np.average(recreated_scores)
        network_sizes_std_scores[size_index] = np.std(recreated_scores, ddof=1)

    if fig_and_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_and_ax
    ax.errorbar(network_sizes, network_sizes_avg_scores, network_sizes_std_scores, linestyle='None', capsize=3, label=option)

    ax.axvline(x=optimised_N_total_nodes, linestyle='--', color="red", alpha=0.6)
    ax.set_xlabel("number of nodes")
    ax.set_ylabel("cross-entropy (bits)")
    ax.legend()
    if log_type:
        ax.set_xscale("log")
        ax.set_yscale("log")
    cross_entropy_network_size_name = str_name_dict["cross_entropy_network_size_name"]
    png_extension = str_name_dict["png_extension"]
    pdf_extension = str_name_dict["pdf_extension"]
    if save_plot:
        fig.savefig(data_directory + cross_entropy_network_size_name + png_extension)
        fig.savefig(data_directory + cross_entropy_network_size_name + pdf_extension)


def get_optimised_parameters(optimised_parameters_location):
    try:
        with open(optimised_parameters_location + "/all_optimised_parameters.json", "r") as configfile:
            parameters = json.load(configfile)
    except IOError:
        print("Could not process the file due to IOError")
        sys.exit(1)
    return parameters


def optimised_convolutive_graph(rng, parameters, in_degree_elements, out_degree_elements, return_positions=False):
    if return_positions:
        graph, positions = convolutive_graph_gen(parameters["m_0_nodes"], parameters["rho_probability"],
                                      parameters["l_cardinality_partitions"], parameters["N_total_nodes"],
                                      parameters["E_k"], parameters["phi_U_probability"],
                                      parameters["phi_D_probability"],
                                      rng, in_degree_elements, out_degree_elements, parameters["dimensions"],
                                      delta=parameters["delta"], noise_factor=parameters["noise_factor"], spatial=True,
                                      pc=parameters["pc"], return_positions=return_positions)
        return graph, positions
    else:
        graph = convolutive_graph_gen(parameters["m_0_nodes"], parameters["rho_probability"],
                                      parameters["l_cardinality_partitions"], parameters["N_total_nodes"],
                                      parameters["E_k"], parameters["phi_U_probability"],
                                      parameters["phi_D_probability"],
                                      rng, in_degree_elements, out_degree_elements, parameters["dimensions"],
                                      delta=parameters["delta"], noise_factor=parameters["noise_factor"], spatial=True,
                                      pc=parameters["pc"])
        return graph


def get_optimised_connectivity(option, N_total_nodes=None, relative_path_to_connectivity=None, int_random_generator=None, return_positions=False):
    neuprint_data_location = 'data_preparation'
    optimised_parameters_location = 'optimised_parameters/' + option
    if relative_path_to_connectivity is not None:
        total_neuprint_data_location = relative_path_to_connectivity + neuprint_data_location
        total_optimised_parameters_location = relative_path_to_connectivity + optimised_parameters_location
    else:
        total_neuprint_data_location = neuprint_data_location
        total_optimised_parameters_location = optimised_parameters_location

    # check if the chosen option already has downloaded data for the dataset,
    # if it does not, it will download the data.
    # (Be aware if only the indegree or only the outdegree has been downloaded,
    # that it will be replaced with new degree data.)
    if not data_downloaded(option, total_neuprint_data_location):
        if option == "manc" or option == "hemibrain":
            print("Data not yet downloaded")
            """
            Look at the quickstart guide on neuprint to get your own token:
            https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html
            and replace the token below with your own 
            """
            token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imlhbi52dmxpZXRAZ21haWwuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKMVpoMnRyWGNUOVgxWWF5NE8zRHExUHBaUjdJVXAzQ1dfLUNVTm1PcmVkdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3OTAxOTY1N30.GnYkuGJjLyxu7lEWqrL9BkxMwLx-7tIM_n63DyLvR9Q'

            extract_neuprint_data(option, token, total_neuprint_data_location)
        elif option == "pyc-pyc":
            get_degrees_from_root_ids_csv_file(total_neuprint_data_location + "/211019_pyc-pyc_subgraph_v185.csv", saved_filename='pyc-pyc',
                                               first_column_index=4)

    # save degree data in variables
    in_degree_elements, out_degree_elements = get_degree_data(option, total_neuprint_data_location)

    parameters = get_optimised_parameters(total_optimised_parameters_location)

    # could get and change parameters e.g.:
    if N_total_nodes is not None:
        parameters["N_total_nodes"] = N_total_nodes
    if int_random_generator is not None:
        parameters["int_random_generator"] = int_random_generator
    # parameters["N_total_nodes"] = 1000
    # parameters["dimensions"] = [2, 0, 2, 0, 2, 0]

    # or could get seed used for optimisation:
    rng = np.random.default_rng(parameters["int_random_generator"])
    if return_positions:
        graph, positions = optimised_convolutive_graph(rng, parameters, in_degree_elements, out_degree_elements,
                                                       return_positions=return_positions)
        return graph, positions
    else:
        graph = optimised_convolutive_graph(rng, parameters, in_degree_elements, out_degree_elements)
        return graph


def create_and_save_model_csv_degree_elements(option, number_of_seeds, total_nodes=None):
    base_folder = "data_preparation/degree_elements/" + option + "/"
    indegree_name = "indegree"
    outdegree_name = "outdegree"
    degrees_model_name = "degrees_model_" + str(total_nodes) + "_"

    for seed_number in range(number_of_seeds):
        # create connectivity
        graph = get_optimised_connectivity(option, int_random_generator=seed_number, N_total_nodes=total_nodes)
        # results from generative convolutional model
        in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix=graph)

        degrees_dataframe = pd.DataFrame({
            indegree_name: in_degree_elements_model,
            outdegree_name: out_degree_elements_model
        })

        filename = f"{degrees_model_name}{seed_number}.csv"
        degrees_dataframe.to_csv(base_folder + filename, index=False)


def get_degree_elements(connectivity_matrix):
    in_degrees, in_degree_counts = np.unique(np.sum(connectivity_matrix, axis=1), return_counts=True)
    out_degrees, out_degree_counts = np.unique(np.sum(connectivity_matrix, axis=0), return_counts=True)
    in_degree_elements_model = np.repeat(in_degrees, in_degree_counts)
    out_degree_elements_model = np.repeat(out_degrees, out_degree_counts)
    return in_degree_elements_model, out_degree_elements_model


def plot_and_save_histograms(option, data_directory, str_name_dict, in_degree_elements, out_degree_elements,
                             in_degree_elements_model, out_degree_elements_model, log_type=True, binsize=20,
                             plot_detailed=True):
    hist_degree_distributions_name = str_name_dict["hist_degree_distributions_name"]
    png_extension = str_name_dict["png_extension"]
    pdf_extension = str_name_dict["pdf_extension"]
    detailed_name = str_name_dict["detailed_name"]
    in_degree_elements_matlab_sim = out_degree_elements_matlab_sim = [0]

    figure, ax = hist_plot_data_model_degree_distributions(option, in_degree_elements, in_degree_elements_model,
                                                           in_degree_elements_matlab_sim, out_degree_elements,
                                                           out_degree_elements_model,
                                                           out_degree_elements_matlab_sim, log_type=log_type, binsize=binsize)
    figure.savefig(data_directory + hist_degree_distributions_name + png_extension)
    figure.savefig(data_directory + hist_degree_distributions_name + pdf_extension)
    if plot_detailed:
        figure.set_size_inches(20, 11.25)

        figure.savefig(data_directory + hist_degree_distributions_name + detailed_name + png_extension)
        figure.savefig(data_directory + hist_degree_distributions_name + detailed_name + pdf_extension)


def score_and_plot(connectivity_matrix, name_type, in_degree_elements, out_degree_elements, option, weight, log_type=True, type_values=None, binsize=20):
    in_degree_elements_matlab_sim = out_degree_elements_matlab_sim = [0]
    if isinstance(connectivity_matrix, np.ndarray):
        in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix)
        figure, ax = hist_plot_data_model_degree_distributions(option, in_degree_elements, in_degree_elements_model,
                                                               in_degree_elements_matlab_sim, out_degree_elements,
                                                               out_degree_elements_model,
                                                               out_degree_elements_matlab_sim, log_type=log_type,
                                                               model_data_label=name_type, binsize=binsize)

        score = elements_linear_cross_entropy(in_degree_elements_model, out_degree_elements_model, in_degree_elements,
                                              out_degree_elements, weight)
        print(name_type + ":", score)
        return figure, score
    elif isinstance(connectivity_matrix, list) and all(isinstance(matrix, np.ndarray) for matrix in connectivity_matrix):
        if len(connectivity_matrix) == 1:
            matrix = connectivity_matrix[0]
            in_degree_elements_model, out_degree_elements_model = get_degree_elements(matrix)
            figure, ax = hist_plot_data_model_degree_distributions(
                option, in_degree_elements, in_degree_elements_model, in_degree_elements_matlab_sim,
                out_degree_elements, out_degree_elements_model, out_degree_elements_matlab_sim,
                log_type=log_type, model_data_label=name_type, binsize=binsize
            )

            score = elements_linear_cross_entropy(in_degree_elements_model, out_degree_elements_model,
                                                  in_degree_elements,
                                                  out_degree_elements, weight)
            print(name_type + ":", score)
            return figure, score
        else:
            if type_values is not None and (len(type_values) == len(connectivity_matrix)):
                zeroth_label = name_type + ": " + str(type_values[0])
                in_degree_elements_model, out_degree_elements_model = get_degree_elements(connectivity_matrix[0])
                figure, ax = hist_plot_data_model_degree_distributions(
                    option, in_degree_elements, in_degree_elements_model, in_degree_elements_matlab_sim,
                    out_degree_elements, out_degree_elements_model, out_degree_elements_matlab_sim,
                    log_type=log_type, model_data_label=zeroth_label, binsize=binsize
                )
                score = elements_linear_cross_entropy(in_degree_elements_model, out_degree_elements_model,
                                                      in_degree_elements,
                                                      out_degree_elements, weight)
                scores = {type_values[0]: score}
                print(name_type + ":", score)
                for label, matrix in enumerate(connectivity_matrix[1:], 1):
                    type_label = name_type + ": " + str(type_values[label])
                    in_degree_elements_model, out_degree_elements_model = get_degree_elements(matrix)
                    hist_additive_plot_degree_distributions(
                        in_degree_elements_model, out_degree_elements_model, figure, ax, type_label, log_type=log_type, binsize=binsize
                    )

                    score = elements_linear_cross_entropy(in_degree_elements_model, out_degree_elements_model, in_degree_elements,
                                                          out_degree_elements, weight)
                    scores[type_values[label]] = score
                    print(name_type + ":", score)
                return figure, scores
            else:
                raise ValueError("The number of values indicating the subtype is not equal to the number of connectivity matrices")
    else:
        raise TypeError("connectivity_matrix should either be a np.ndarray or a list of np.ndarray ")


def get_bin_edges(list_elements, binsize=20):
    max_value = np.max(np.concatenate(list_elements))
    bin_edges = range(0, max_value + binsize, binsize)
    return bin_edges


def plot_degree_counts(connectivity_graph, title="Convolutive model"):
    fig_indegrees, ax_indegrees = plt.subplots()
    indegrees, counts = np.unique(np.sum(connectivity_graph, axis=1), return_counts=True)
    probs = counts/np.sum(counts)
    # print(indegrees)
    ax_indegrees.bar(indegrees, probs, align='center')
    ax_indegrees.set_title(title)
    ax_indegrees.set_xlabel("indegree")
    ax_indegrees.set_ylabel("probability")

    fig_outdegrees, ax_outdegrees = plt.subplots()
    outdegrees, counts = np.unique(np.sum(connectivity_graph, axis=0), return_counts=True)
    probs = counts / np.sum(counts)
    ax_outdegrees.bar(outdegrees, probs, align='center')
    ax_outdegrees.set_title(title)
    ax_outdegrees.set_xlabel("outdegree")
    ax_outdegrees.set_ylabel("probability")


def hist_plot_data_model_degree_distributions(option, in_degree_elements, in_degree_elements_model,
                                              in_degree_elements_matlab_sim, out_degree_elements,
                                              out_degree_elements_model, out_degree_elements_matlab_sim, log_type=False, model_data_label='sim', binsize=20):
    fig, ax = plt.subplots(1, 2)

    bin_edges_indegree = get_bin_edges([in_degree_elements, in_degree_elements_model, in_degree_elements_matlab_sim], binsize=binsize)
    bin_edges_outdegree = get_bin_edges([out_degree_elements, out_degree_elements_model, out_degree_elements_matlab_sim], binsize=binsize)
    density_type = True

    ax[0].hist(in_degree_elements, label='data', alpha=1, bins=bin_edges_indegree, density=density_type, edgecolor='black',
               linewidth=1, log=log_type)
    ax[0].hist(in_degree_elements_model, label=model_data_label, alpha=0.8, bins=bin_edges_indegree, density=density_type,
               linewidth=2, histtype='step', log=log_type)
    if option == "matlab_data":
        ax[0].hist(in_degree_elements_matlab_sim, label='matlab sim', alpha=0.8, bins=bin_edges_indegree, density=density_type,
                   edgecolor='green', linewidth=2, histtype='step', log=log_type)
    ax[0].set_xlabel("indegree")

    ax[0].set_ylabel("probability")
    ax[1].hist(out_degree_elements, label='data', alpha=1, bins=bin_edges_outdegree, density=density_type, edgecolor='black',
               linewidth=1, log=log_type)
    ax[1].hist(out_degree_elements_model, label=model_data_label, alpha=0.8, bins=bin_edges_outdegree, density=density_type,
               linewidth=2, histtype='step', log=log_type)
    if option == "matlab_data":
        ax[1].hist(out_degree_elements_matlab_sim, label='matlab sim', alpha=0.8, bins=bin_edges_outdegree, density=density_type,
                   edgecolor='green', linewidth=2, histtype='step', log=log_type)
    ax[1].set_xlabel("outdegree")
    if log_type:
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
    ax[1].set_ylabel("probability")
    if not log_type:
        ax[0].ticklabel_format(axis='y', scilimits=(-2, 2))
        ax[1].ticklabel_format(axis='y', scilimits=(-2, 2))
    fig.tight_layout()
    ax[0].legend()
    ax[1].legend()
    return fig, ax


def hist_additive_plot_degree_distributions(in_degree_elements_model, out_degree_elements_model, fig, ax, label, log_type=False, binsize=20):
    bin_edges_indegree = get_bin_edges([in_degree_elements_model], binsize=binsize)
    bin_edges_outdegree = get_bin_edges([out_degree_elements_model], binsize=binsize)
    density_type = True

    ax[0].hist(in_degree_elements_model, label=label, alpha=0.8, bins=bin_edges_indegree, density=density_type,
               linewidth=2, histtype='step', log=log_type)
    ax[1].hist(out_degree_elements_model, label=label, alpha=0.8, bins=bin_edges_outdegree, density=density_type,
               linewidth=2, histtype='step', log=log_type)
    ax[0].legend()
    ax[1].legend()
    return fig


def step_plot_data_model_degree_distributions(option, in_degree, in_degree_gen_model, in_degree_online_probs,
                                              out_degree, out_degree_gen_model, out_degree_online_probs):
    fig, ax = plt.subplots(1, 2)
    ax[0].step(np.arange(len(in_degree)), in_degree, label="data", where='mid')
    ax[0].step(np.arange(len(in_degree_gen_model)), in_degree_gen_model, label="own generative model", where='mid')
    if option == "C_elegans":
        ax[0].step(np.arange(len(in_degree_online_probs)), in_degree_online_probs, label="online simulator",
                   where='mid')
    ax[0].legend()
    ax[0].set_xlabel("indegree")
    ax[0].set_ylabel("probability")

    ax[1].step(np.arange(len(out_degree)), out_degree, label="data", where='mid')
    ax[1].step(np.arange(len(out_degree_gen_model)), out_degree_gen_model, label="own generative model", where='mid')
    if option == "C_elegans":
        ax[1].step(np.arange(len(out_degree_online_probs)), out_degree_online_probs, label="online simulator",
                   where='mid')

    ax[1].set_xlabel("outdegree")
    ax[1].set_ylabel("probability")
    ax[1].legend()
    fig.tight_layout()


def save_png_pdf(figure, directory, filename):
    png_extension = ".png"
    pdf_extension = ".pdf"
    figure.savefig(directory + filename + png_extension)
    figure.savefig(directory + filename + pdf_extension)