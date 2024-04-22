import numpy as np
from neuprint import Client
from neuprint import fetch_roi_hierarchy
from neuprint import fetch_neurons, fetch_adjacencies, connection_table_to_matrix, NeuronCriteria as NC
from neuprint import fetch_traced_adjacencies
from neuprint import merge_neuron_properties
from data_prep_utils import *

"""
neuPrint: An open access tool for EM connectomics.
Plaza SM, Clements J, Dolafi T, Umayam L, Neubarth NN, Scheffer LK and Berg S
Front. Neuroinform. 16:896292. doi: 10.3389/fninf.2022.896292 (https://doi.org/10.1101/2020.01.16.909465)
This work is licensed under a Creative Commons Attribution 4.0 International License
(http://creativecommons.org/licenses/by/4.0/).
"""
# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
# option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
# option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"

# Look at the quickstart guide on neuprint to get your own token:
# https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html
# token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imlhbi52dmxpZXRAZ21haWwuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKMVpoMnRyWGNUOVgxWWF5NE8zRHExUHBaUjdJVXAzQ1dfLUNVTm1PcmVkdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3OTAxOTY1N30.GnYkuGJjLyxu7lEWqrL9BkxMwLx-7tIM_n63DyLvR9Q'

# graph = extract_neuprint_data(option, token)

# from MICrONS Explorer https://www.microns-explorer.org/phase1, raw data at: https://zenodo.org/records/5579388
# accompanying paper: Turner, N. L., Macrina, T., Bae, J. A., Yang, R., Wilson, A. M., Schneider-Mizell, C., Lee, K.,
# Lu, R., Wu, J., Bodor, A. L., Bleckert, A. A., Brittain, D., Froudarakis, E., Dorkenwald, S., Collman, F., Kemnitz,
# N., Ih, D., Silversmith, W. M., Zung, J., … Seung, H. S. (2022). Reconstruction of neocortex: Organelles,
# compartments, cells, circuits, and activity. Cell, 185(6), 1082-1100.e24. https://doi.org/10.1016/j.cell.2022.01.023
graph = get_degrees_from_root_ids_csv_file("211019_pyc-pyc_subgraph_v185.csv", saved_filename='pyc-pyc', first_column_index=4)

# fig, ax = plt.subplots()
# ax.imshow(graph[:, np.argsort(graph.sum(axis=0))][np.argsort(graph.sum(axis=1)), :])
# plt.show()
