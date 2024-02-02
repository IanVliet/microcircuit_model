import numpy as np
from neuprint import Client
from neuprint import fetch_roi_hierarchy
from neuprint import fetch_neurons, fetch_adjacencies, connection_table_to_matrix, NeuronCriteria as NC
from neuprint import fetch_traced_adjacencies
from neuprint import merge_neuron_properties
from data_prep_utils import *

# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
# option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"

# Look at the quickstart guide on neuprint to get your own token:
# https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html
token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imlhbi52dmxpZXRAZ21haWwuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKMVpoMnRyWGNUOVgxWWF5NE8zRHExUHBaUjdJVXAzQ1dfLUNVTm1PcmVkdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3OTAxOTY1N30.GnYkuGJjLyxu7lEWqrL9BkxMwLx-7tIM_n63DyLvR9Q'

extract_neuprint_data(option, token)
