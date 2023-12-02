import numpy as np
from neuprint import Client
from neuprint import fetch_roi_hierarchy
from neuprint import fetch_neurons, fetch_adjacencies, connection_table_to_matrix, NeuronCriteria as NC
from neuprint import fetch_traced_adjacencies
from neuprint import merge_neuron_properties

# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
# option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"

if option == "manc":
    client = Client('neuprint.janelia.org', dataset='manc:v1.0', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imlhbi52dmxpZXRAZ21haWwuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKMVpoMnRyWGNUOVgxWWF5NE8zRHExUHBaUjdJVXAzQ1dfLUNVTm1PcmVkdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3OTAxOTY1N30.GnYkuGJjLyxu7lEWqrL9BkxMwLx-7tIM_n63DyLvR9Q')
    traced_df, roi_conn_df = fetch_traced_adjacencies('manc-traced-adjacencies-v1.0')
elif option == "hemibrain":
    client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1',
                    token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imlhbi52dmxpZXRAZ21haWwuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKMVpoMnRyWGNUOVgxWWF5NE8zRHExUHBaUjdJVXAzQ1dfLUNVTm1PcmVkdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3OTAxOTY1N30.GnYkuGJjLyxu7lEWqrL9BkxMwLx-7tIM_n63DyLvR9Q')
    traced_df, roi_conn_df = fetch_traced_adjacencies('hemibrain-traced-adjacencies-v1.2.1')
else:
    print("The chosen option", option, "is not one of the possible options.")
    exit(1)
bodyIds_pre, indices_pre, outdegrees = np.unique(roi_conn_df.bodyId_pre, return_index=True, return_counts=True)
bodyIds_post, indices_post, indegrees = np.unique(roi_conn_df.bodyId_post, return_index=True, return_counts=True)

print(outdegrees)
print(indegrees)

if option == "manc":
    with open('manc_v1.0_indegrees.npy', 'wb') as indegree_file:
        np.save(indegree_file, indegrees)

    with open('manc_v1.0_outdegrees.npy', 'wb') as outdegree_file:
        np.save(outdegree_file, outdegrees)
elif option == "hemibrain":
    with open('hemibrain_v1.2.1_indegrees.npy', 'wb') as indegree_file:
        np.save(indegree_file, indegrees)

    with open('hemibrain_v1.2.1_outdegrees.npy', 'wb') as outdegree_file:
        np.save(outdegree_file, outdegrees)
else:
    print("The chosen option", option, "is not one of the possible options.")
    exit(1)

