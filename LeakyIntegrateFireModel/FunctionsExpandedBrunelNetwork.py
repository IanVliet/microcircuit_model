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


# generates BARABÁSI AND ALBERT (BA) graph from code in MATLAB shared in "Graph-theoretical derivation of brain
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
    connectivity_graph = np.zeros((num_cells, num_cells))
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
                          dimensions, pc=300, delta=3, noise_factor=1.5, spatial=True):

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
    if spatial:
        A_1 = spatial_block_ba_graph(N_1, dimensions[0], dimensions[1], dimensions[2], dimensions[3],
                                     dimensions[4], dimensions[5], delta, noise_factor, m_0_nodes, rng,
                                     sigma, rho_probability)[0]
        A_2 = spatial_block_ba_graph(N_2, dimensions[0], dimensions[1], dimensions[2], dimensions[3],
                                     dimensions[4], dimensions[5], delta, noise_factor, m_0_nodes, rng,
                                     sigma, rho_probability)[0]
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
        in_degrees, in_degree_counts = np.unique(np.sum(A_1, axis=1), return_counts=True)
        out_degrees, out_degree_counts = np.unique(np.sum(A_1, axis=0), return_counts=True)
        in_degree_elements_model = np.repeat(in_degrees, in_degree_counts)
        out_degree_elements_model = np.repeat(out_degrees, out_degree_counts)
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
    return B


# NEEDS TO BE FIXED STILL!!! --> seems impossible to calculate
def convolutive_probabilities(in_degree_distribution, out_degree_distribution, N_total_nodes,
                              l_cardinality_partitions, p_probability, phi_U_probability, phi_D_probability):
    M = round(N_total_nodes/l_cardinality_partitions)
    print(M)
    in_degree_model = []
    out_degree_model = []
    delta_0_distribution = np.zeros(N_total_nodes)
    delta_0_distribution[0] = 1
    range_to_N = np.arange(N_total_nodes, dtype=float)
    # NEEDS TO BE FIXED STILL
    binomial_phi_U_distribution = binom.pmf(range_to_N, l_cardinality_partitions, phi_U_probability)
    binomial_phi_D_distribution = binom.pmf(range_to_N, l_cardinality_partitions, phi_D_probability)
    print(comb(l_cardinality_partitions, range_to_N))
    print(binomial_phi_U_distribution)
    print(binomial_phi_D_distribution)
    for k in range(1, N_total_nodes+1):
        left_part = (1-p_probability)*delta_0_distribution[:k] + p_probability*binomial_phi_U_distribution[:k]
        right_part = p_probability*delta_0_distribution[:k] + (1-p_probability)*binomial_phi_D_distribution[:k]
        first_convolution = np.convolve(left_part, right_part)
        power_convolutions = convolve_power(first_convolution, M)
        in_degree_model.append(np.convolve(in_degree_distribution[:k], power_convolutions))
        out_degree_model.append(np.convolve(out_degree_distribution[:k], power_convolutions))
    return in_degree_model, out_degree_model


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


def cross_entropy(data_distribution, model_distribution):
    return -np.sum(data_distribution*np.log2(model_distribution[:len(data_distribution)]))


def linear_combination_cross_entropy(in_degree_cross_entropy, out_degree_cross_entropy, weight):
    return weight * in_degree_cross_entropy + (1 - weight) * out_degree_cross_entropy


def extend_with_zeros(total_length, original_distribution):
    new_distribution = np.zeros(total_length)
    new_distribution[:len(original_distribution)] = original_distribution
    return new_distribution


def produce_connectivity_calculate_cross_entropy(N_total_nodes, int_random_generator, weight, in_degree_elements, out_degree_elements, config):
    rng = np.random.default_rng(int_random_generator)
    m_0_nodes = config["m_0"]
    rho_probability = config["rho"]  # not present in web application? (Noise factor?, Delta?)
    l_cardinality_partitions = config["l"]  # Likely L in the web application
    E_k = config["E_k"]  # likely EK in the web application
    phi_U_probability = config["phi_U"]  # likely phi_U in the web application
    phi_D_probability = config["phi_D"]  # likely phi_D in the web application
    delta = config["delta"]
    noise_factor = config["noise_factor"]
    dimensions = config["dimensions"]
    pc = config["pc"]

    B = convolutive_graph_gen(m_0_nodes, rho_probability, l_cardinality_partitions,
                              N_total_nodes, E_k, phi_U_probability,
                              phi_D_probability, rng, in_degree_elements, out_degree_elements, dimensions,
                              delta=delta, noise_factor=noise_factor, spatial=True, pc=pc)

    # degree distributions of data
    in_degree_values, in_degree_distribution, out_degree_values, out_degree_distribution = \
        get_degree_distributions(in_degree_elements, out_degree_elements)

    in_degree, out_degree = \
        get_interpolated_degree_distributions(in_degree_values, in_degree_distribution, out_degree_values,
                                              out_degree_distribution, np.max(in_degree_elements),
                                              np.max(out_degree_elements))

    # degree distributions of model
    in_degrees, in_degree_counts = np.unique(np.sum(B, axis=1), return_counts=True)
    out_degrees, out_degree_counts = np.unique(np.sum(B, axis=0), return_counts=True)
    in_degree_elements_model = np.repeat(in_degrees, in_degree_counts)
    out_degree_elements_model = np.repeat(out_degrees, out_degree_counts)

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

    return {"score": linear_combination_cross_entropy(in_degree_cross_entropy, out_degree_cross_entropy, weight=weight)}


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
                                              out_degree_elements_model, out_degree_elements_matlab_sim):
    fig, ax = plt.subplots(1, 2)
    num_bins = len(np.unique(in_degree_elements))
    binsize = 20
    max_value_indegree = np.max(np.concatenate([in_degree_elements, in_degree_elements_model, in_degree_elements_matlab_sim]))
    max_value_outdegree = np.max(np.concatenate([out_degree_elements, out_degree_elements_model, out_degree_elements_matlab_sim]))
    bin_edges_indegree = range(0, max_value_indegree + binsize, binsize)
    bin_edges_outdegree = range(0, max_value_outdegree + binsize, binsize)
    density_type = True
    # if num_bins > 100:
    #     num_bins = 100
    ax[0].hist(in_degree_elements, label='data', alpha=1, bins=bin_edges_indegree, density=density_type, edgecolor='black',
               linewidth=1)
    ax[0].hist(in_degree_elements_model, label='sim', alpha=0.8, bins=bin_edges_indegree, density=density_type, edgecolor='red',
               linewidth=2, histtype='step')
    if option == "matlab_data":
        ax[0].hist(in_degree_elements_matlab_sim, label='matlab sim', alpha=0.8, bins=bin_edges_indegree, density=density_type,
                   edgecolor='green', linewidth=2, histtype='step')
    ax[0].set_xlabel("indegree")
    ax[0].set_ylabel("probability")
    ax[1].hist(out_degree_elements, label='data', alpha=1, bins=bin_edges_outdegree, density=density_type, edgecolor='black',
               linewidth=1)
    ax[1].hist(out_degree_elements_model, label='sim', alpha=0.8, bins=bin_edges_outdegree, density=density_type, edgecolor='red',
               linewidth=2, histtype='step')
    if option == "matlab_data":
        ax[1].hist(out_degree_elements_matlab_sim, label='matlab sim', alpha=0.8, bins=bin_edges_outdegree, density=density_type,
                   edgecolor='green', linewidth=2, histtype='step')
    ax[1].set_xlabel("outdegree")
    ax[1].set_ylabel("probability")
    ax[0].ticklabel_format(axis='y', scilimits=(-2, 2))
    ax[1].ticklabel_format(axis='y', scilimits=(-2, 2))
    fig.tight_layout()
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
