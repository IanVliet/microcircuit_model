import numpy as np
from neuprint import Client
from neuprint import fetch_roi_hierarchy
from neuprint import fetch_neurons, fetch_adjacencies, connection_table_to_matrix, NeuronCriteria as NC
from neuprint import fetch_traced_adjacencies
from neuprint import merge_neuron_properties
from leaky_integrate_fire_model.connectivity.connectivity_utils import *
import matplotlib.pyplot as plt
import scipy.io


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
    bodyIds_pre, indices_pre, outdegrees = np.unique(roi_conn_df.bodyId_pre, return_index=True, return_counts=True)
    bodyIds_post, indices_post, indegrees = np.unique(roi_conn_df.bodyId_post, return_index=True, return_counts=True)

    print(outdegrees)
    print(indegrees)
    if data_location is not None:
        data_directory = data_location + "/"
    else:
        data_directory = ""
    if option == "manc":
        with open(data_directory + 'manc_v1.0_indegrees.npy', 'wb') as indegree_file:
            np.save(indegree_file, indegrees)

        with open(data_directory + 'manc_v1.0_outdegrees.npy', 'wb') as outdegree_file:
            np.save(outdegree_file, outdegrees)
    elif option == "hemibrain":
        with open(data_directory + 'hemibrain_v1.2.1_indegrees.npy', 'wb') as indegree_file:
            np.save(indegree_file, indegrees)

        with open(data_directory + 'hemibrain_v1.2.1_outdegrees.npy', 'wb') as outdegree_file:
            np.save(outdegree_file, outdegrees)
    else:
        print("The chosen option", option, "is not one of the possible options.")
        exit(1)


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
