import numpy as np
from neuprint import Client
from neuprint import fetch_roi_hierarchy
from neuprint import fetch_neurons, fetch_adjacencies, connection_table_to_matrix, NeuronCriteria as NC
from neuprint import fetch_traced_adjacencies
from neuprint import merge_neuron_properties

# manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
client = Client('neuprint.janelia.org', dataset='manc:v1.0', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imlhbi52dmxpZXRAZ21haWwuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKMVpoMnRyWGNUOVgxWWF5NE8zRHExUHBaUjdJVXAzQ1dfLUNVTm1PcmVkdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3OTAxOTY1N30.GnYkuGJjLyxu7lEWqrL9BkxMwLx-7tIM_n63DyLvR9Q')
# print(client.fetch_version())
# query = """\
#     MATCH (n :Neuron {`ADMN(L)`: true})
#     WHERE n.pre > 10
#     RETURN n.bodyId AS bodyId, n.type as type, n.instance AS instance, n.pre AS numpre, n.post AS numpost
#     ORDER BY n.pre + n.post DESC
# """
# results = client.fetch_custom(query)
# print(f"Found {len(results)} results")
# print(fetch_roi_hierarchy(False, mark_primary=True, format='text'))
# print(client.fetch_roi_connectivity())
# criteria = NC(rois=['ANm',
#                     # 'ADMN(L)', 'ADMN(R)',  'AbN1(L)', 'AbN1(R)', 'AbN2(L)'
#                     ], status='Traced')
# second_criteria = NC(rois=['ANm',
#                            # 'AbN2(R)', 'AbN3(L)', 'AbN3(R)', 'AbN4(L)',
#                            #  'AbN4(R)', 'AbNT'
#                            ], status='Traced')
# neuron_df, conn_df = fetch_adjacencies(criteria, criteria)

# bodyIds_pre, indices_pre, out_degrees = np.unique(conn_df.bodyId_pre, return_index=True, return_counts=True)
# bodyIds_post, indices_post, in_degrees = np.unique(conn_df.bodyId_post, return_index=True, return_counts=True)
# for bodyId_pre_counter, bodyId_post_counter, conn_weight in zip(conn_df.bodyId_pre, conn_df.bodyId_post, conn_df.weight):
#     print((bodyId_pre_counter, bodyId_post_counter, conn_weight))

traced_df, roi_conn_df = fetch_traced_adjacencies('manc-traced-adjacencies-v1.0')
bodyIds_pre, indices_pre, outdegrees = np.unique(roi_conn_df.bodyId_pre, return_index=True, return_counts=True)
bodyIds_post, indices_post, indegrees = np.unique(roi_conn_df.bodyId_post, return_index=True, return_counts=True)
# bodyIds_pre = bodyIds_pre.astype(float)
# bodyIds_post = bodyIds_post.astype(float)
# combined_Ids = np.concatenate(bodyIds_pre, bodyIds_post)
# num_cells = len(np.unique(combined_Ids))
print(outdegrees)
print(indegrees)
# with open('manc_v1.0_N_' + str(num_cells) + '_indegrees.npy', 'wb') as indegree_file:
#     np.save(indegree_file, indegrees)
#
# with open('manc_v1.0_N_' + str(num_cells) + '_outdegrees.npy', 'wb') as outdegree_file:
#     np.save(outdegree_file, outdegrees)
with open('manc_v1.0_indegrees.npy', 'wb') as indegree_file:
    np.save(indegree_file, indegrees)

with open('manc_v1.0_outdegrees.npy', 'wb') as outdegree_file:
    np.save(outdegree_file, outdegrees)
