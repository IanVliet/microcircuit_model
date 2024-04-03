import itertools

import numpy as np
from neuprint import Client
from neuprint import fetch_roi_hierarchy
from neuprint import fetch_neurons, fetch_adjacencies, connection_table_to_matrix, NeuronCriteria as NC
from neuprint import fetch_traced_adjacencies
from neuprint import merge_neuron_properties
import dask.dataframe as dd
from leaky_integrate_fire_model.connectivity.connectivity_utils import *
import matplotlib.pyplot as plt
import scipy.io
import os
import pandas as pd


def data_downloaded(option, data_location=None):
    if data_location is not None:
        data_directory = data_location + "/"
        if not os.path.exists(data_location):
            os.makedirs(data_location)
    else:
        data_directory = ""
    if option == "manc":
        indegree = os.path.isfile(data_directory + 'manc_v1.0_indegrees.npy')
        outdegree = os.path.isfile(data_directory + 'manc_v1.0_outdegrees.npy')

    elif option == "hemibrain":
        indegree = os.path.isfile(data_directory + 'hemibrain_v1.2.1_indegrees.npy')
        outdegree = os.path.isfile(data_directory + 'hemibrain_v1.2.1_outdegrees.npy')

    elif option == "pyc-pyc":
        indegree = os.path.isfile(data_directory + 'pyc-pyc_indegrees.npy')
        outdegree = os.path.isfile(data_directory + 'pyc-pyc_outdegrees.npy')
    else:
        print("The chosen option", option, "is not one of the possible options.")
        exit(1)
    return indegree and outdegree


def extract_neuprint_data(option, token, data_location=None):
    """
    Extracts raw data from the neuprint website, and saves it as indegree and outdegree data. (.npy format)
    neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
    manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
    hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"
    """
    if option == "manc":
        client = Client('neuprint.janelia.org', dataset='manc:v1.0', token=token)
        traced_df, roi_conn_df = fetch_traced_adjacencies('../../manc-traced-adjacencies-v1.0')
    elif option == "hemibrain":
        client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1',
                        token=token)
        traced_df, roi_conn_df = fetch_traced_adjacencies('../../hemibrain-traced-adjacencies-v1.2.1')
    else:
        print("The chosen option ", option, "is not one of the possible options.")
        exit(1)
    unique_connections = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)
    total_conn_df = unique_connections['weight'].sum()
    count_duplicate_connections(total_conn_df.bodyId_pre, total_conn_df.bodyId_post)

    bodyIds_pre, indices_pre, outdegrees = np.unique(total_conn_df.bodyId_pre, return_index=True, return_counts=True)
    bodyIds_post, indices_post, indegrees = np.unique(total_conn_df.bodyId_post, return_index=True, return_counts=True)

    print("indegrees:", np.sort(indegrees), len(indegrees), sum(indegrees))
    print("outdegrees:", np.sort(outdegrees), len(outdegrees), sum(outdegrees))

    graph = get_connectivity_graph_from_pre_post_ids(total_conn_df.bodyId_pre, total_conn_df.bodyId_post)
    indegrees_graph, outdegrees_graph = get_degree_elements(graph)
    print("indegrees graph:", np.sort(indegrees_graph), len(indegrees_graph), sum(indegrees_graph))
    print("outdegrees graph:", np.sort(outdegrees_graph), len(outdegrees_graph), sum(outdegrees_graph))

    if data_location is not None:
        data_directory = data_location + "/"
    else:
        data_directory = ""
    if option == "manc":
        with open(data_directory + 'manc_v1.0_indegrees.npy', 'wb') as indegree_file:
            np.save(indegree_file, indegrees_graph)

        with open(data_directory + 'manc_v1.0_outdegrees.npy', 'wb') as outdegree_file:
            np.save(outdegree_file, outdegrees_graph)
    elif option == "hemibrain":
        with open(data_directory + 'hemibrain_v1.2.1_indegrees.npy', 'wb') as indegree_file:
            np.save(indegree_file, indegrees_graph)

        with open(data_directory + 'hemibrain_v1.2.1_outdegrees.npy', 'wb') as outdegree_file:
            np.save(outdegree_file, outdegrees_graph)
    else:
        print("The chosen option", option, "is not one of the possible options.")
        exit(1)
    return graph


def convert_npy_to_mat(in_degree_filename, out_degree_filename, mat_filename):
    """
    Converts .npy in-degree and out-degree files to a .mat file with the data under "din" and "dout", respectively.
    The filenames should include the extensions.
    """
    with open(in_degree_filename, 'rb') as indegree_file:
        in_degree_elements = np.load(indegree_file)

    with open(out_degree_filename, 'rb') as outdegree_file:
        out_degree_elements = np.load(outdegree_file)

    scipy.io.savemat(mat_filename,
                     dict(din=in_degree_elements.astype(float), dout=out_degree_elements.astype(float)))


def to_txt_distributions(in_degree_elements, out_degree_elements):
    in_degree_values, in_degree_distribution, out_degree_values, out_degree_distribution = \
        get_degree_distributions(in_degree_elements, out_degree_elements)
    lfig, ax = plt.subplots()
    ax.plot(in_degree_distribution)
    with open("indegree_distribution_matlab_data.txt", 'w') as in_degree_file:
        for in_degree_count, in_degree_prob in enumerate(in_degree_distribution):
            in_degree_file.write(str(in_degree_count) + "," + str(in_degree_prob))
            in_degree_file.write("\n")

    with open("outdegree_distribution_matlab_data.txt", 'w') as out_degree_file:
        for out_degree_count, out_degree_prob in enumerate(out_degree_distribution):
            out_degree_file.write(str(out_degree_count) + "," + str(out_degree_prob))
            out_degree_file.write("\n")
    plt.show()


def to_csv_degree_elements(in_degree_elements, out_degree_elements, option):

    indegree_name = "indegree"
    outdegree_name = "outdegree"

    experimental_degree_data_filename = "data_preparation/degree_elements/degrees_experimental_" + str(option) + ".csv"
    if not os.path.isfile(experimental_degree_data_filename):
        degrees_experimental_dataframe = pd.DataFrame({
            indegree_name: in_degree_elements,
            outdegree_name: out_degree_elements
        })
        degrees_experimental_dataframe.to_csv(experimental_degree_data_filename, index=False)
    

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


def get_degrees_from_root_ids_csv_file(filename, saved_filename, data_location=None, first_column_index=4):
    # get the columns from the csv file
    root_ids = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(first_column_index, first_column_index+1))

    # 'pre_root_id' is the fifth column and 'post_root_id' is the sixth column (indexing starts at 0)
    pre_root_ids = root_ids[:, 0]
    post_root_ids = root_ids[:, 1]

    unique_pre_root_ids, unique_post_root_ids = count_duplicate_connections(pre_root_ids, post_root_ids)

    pre_root_id, indices_pre, outdegrees = np.unique(unique_pre_root_ids, return_index=True, return_counts=True)
    post_root_id, indices_post, indegrees = np.unique(unique_post_root_ids, return_index=True, return_counts=True)

    print("indegrees:", indegrees, len(indegrees), sum(indegrees))
    print("outdegrees:", outdegrees, len(outdegrees), sum(outdegrees))

    graph = get_connectivity_graph_from_pre_post_ids(pre_root_ids, post_root_ids)
    indegrees_graph, outdegrees_graph = get_degree_elements(graph)
    print("indegrees graph:", indegrees_graph, len(indegrees_graph), sum(indegrees_graph))
    print("outdegrees graph:", outdegrees_graph, len(outdegrees_graph), sum(outdegrees_graph))

    if data_location is not None:
        data_directory = data_location + "/"
    else:
        data_directory = ""

    with open(data_directory + saved_filename + '_indegrees.npy', 'wb') as indegree_file:
        np.save(indegree_file, indegrees_graph)

    with open(data_directory + saved_filename + '_outdegrees.npy', 'wb') as outdegree_file:
        np.save(outdegree_file, outdegrees_graph)

    return graph


def get_degrees_from_root_ids_parquet_file(file_directory, saved_filename, data_location=None):
    # get the dataframe from the parquet file
    synapse_graph = dd.read_parquet(file_directory)

    # Should filter the neurons to only have a proofreading status of 'extended' or atleast understand why the number of
    # neurons is 131730 instead of the 70000 neurons mentioned for the dataset.

    # get the presynaptic and postsynaptic columns and save as dask Arrays
    pre_pt_root_ids = synapse_graph['pre_pt_root_id'].values.compute()
    post_pt_root_ids = synapse_graph['post_pt_root_id'].values.compute()

    unique_pre_root_ids, unique_post_root_ids = count_duplicate_connections(pre_pt_root_ids, post_pt_root_ids)

    pre_root_id, indices_pre, outdegrees = np.unique(unique_pre_root_ids, return_index=True, return_counts=True)
    post_root_id, indices_post, indegrees = np.unique(unique_post_root_ids, return_index=True, return_counts=True)

    print("indegrees:", indegrees, len(indegrees), sum(indegrees))
    print("outdegrees:", outdegrees, len(outdegrees), sum(outdegrees))
    #
    graph = get_connectivity_graph_from_pre_post_ids(pre_pt_root_ids, post_pt_root_ids)
    indegrees_graph, outdegrees_graph = get_degree_elements(graph)
    print("indegrees graph:", indegrees_graph, len(indegrees_graph), sum(indegrees_graph))
    print("outdegrees graph:", outdegrees_graph, len(outdegrees_graph), sum(outdegrees_graph))

    if data_location is not None:
        data_directory = data_location + "/"
    else:
        data_directory = ""

    with open(data_directory + saved_filename + '_indegrees.npy', 'wb') as indegree_file:
        np.save(indegree_file, indegrees)

    with open(data_directory + saved_filename + '_outdegrees.npy', 'wb') as outdegree_file:
        np.save(outdegree_file, outdegrees)

    return graph


def filter_out_non_extended_neurons(unfiltered_graph, proofreading_dataframe):
    """Untested function, it should expect the proofreading_dataframe as a pandas dataframe, and the unfiltered graph as
    a dask dataframe. (although using .compute() should also be possible using a dask dataframe for the
    proofreading_dataframe). """
    proofreading_dataframe.columns = ['id', 'valid', 'pt_position_x', 'pt_position_y', 'pt_position_z',
                                      'pt_supervoxel_id', 'pt_root_id', 'valid_id', 'status_dendrite', 'status_axon']
    extended_neurons = proofreading_dataframe[(proofreading_dataframe["status_dendrite"] == "extended" or
                                               proofreading_dataframe["status_dendrite"] == "clean") and
                                              (proofreading_dataframe["status_axon"] == "extended" or
                                               proofreading_dataframe["status_axon"] == "clean")]
    extended_neurons_proofreading_checked = extended_neurons[extended_neurons["valid_id"] ==
                                                             extended_neurons["pt_root_id"]]
    filter_values = extended_neurons_proofreading_checked["valid_id"].tolist()
    filtered_edgelist_pre_only = unfiltered_graph[unfiltered_graph["pre_pt_root_id"].isin(filter_values)]
    filtered_edgelist = filtered_edgelist_pre_only[filtered_edgelist_pre_only["post_pt_root_id"].isin(filter_values)]
    filtered_graph = filtered_edgelist.compute()
    return filtered_graph


def count_duplicate_connections(pre_root_ids, post_root_ids):
    connections_array = np.array((pre_root_ids, post_root_ids)).T
    unique_connections = np.unique(connections_array, axis=0)
    print("number of connections (duplicates not explicitly removed):", len(connections_array))
    print("number of connections (duplicates removed):", len(unique_connections))
    unique_pre_root_ids, unique_post_root_ids = np.split(unique_connections, 2, axis=1)
    return unique_pre_root_ids, unique_post_root_ids


def get_connectivity_graph_from_pre_post_ids(pre_root_ids, post_root_ids):
    pre_root_ids = list(pre_root_ids)
    post_root_ids = list(post_root_ids)
    neuron_ids_set = set(pre_root_ids + post_root_ids)
    number_of_neurons = len(neuron_ids_set)
    neuron_ids_dict = {neuron_id: neuron_index for neuron_index, neuron_id in enumerate(neuron_ids_set)}

    graph = np.zeros((number_of_neurons, number_of_neurons), dtype=bool)

    for pre_id, post_id in zip(pre_root_ids, post_root_ids):
        graph[neuron_ids_dict.get(post_id), neuron_ids_dict.get(pre_id)] = 1
    return graph


def save_large_synapse_graph_csv_as_parquet_files():
    # If you have not yet downloaded the synapse graph it should be directly downloadable through the following link
    # https://s3.amazonaws.com/bossdb-open-data/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv
    # It should also be possible to directly load it into python, although the server might disconnect (for reasons unknown)
    # e.g. use synapse_graph = dd.read_csv("https://s3.amazonaws.com/bossdb-open-data/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv")
    synapse_graph = dd.read_csv("../../../degree_data/synapses_pni_2.csv")
    synapse_graph.columns = ['id', 'valid', 'pre_pt_position_x', 'pre_pt_position_y', 'pre_pt_position_z',
                             'post_pt_position_x', 'post_pt_position_y', 'post_pt_position_z',
                             'ctr_pt_position_x', 'ctr_pt_position_y', 'ctr_pt_position_z',
                             'pre_pt_supervoxel_id', 'post_pt_supervoxel_id', 'pre_pt_root_id', 'post_pt_root_id',
                             'size']
    synapse_graph.to_parquet('cortical_microns/', name_function=lambda x: f"synapse_graph_part-{x}.parquet")
