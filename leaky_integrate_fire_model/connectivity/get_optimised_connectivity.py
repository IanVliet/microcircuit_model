from data_preparation.data_prep_utils import *

# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
# option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"


neuprint_data_location = 'data_preparation'
optimised_parameters_location = 'optimised_parameters/' + option
# check if the chosen option already has downloaded data for the dataset,
# if it does not, it will download the data.
# (Be aware if only the indegree or only the outdegree has been downloaded,
# that it will be replaced with new degree data.)
if not data_downloaded(option, neuprint_data_location):
    print("Data not yet downloaded")
    """
    Look at the quickstart guide on neuprint to get your own token:
    https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html
    and replace the token below with your own 
    """
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imlhbi52dmxpZXRAZ21haWwuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKMVpoMnRyWGNUOVgxWWF5NE8zRHExUHBaUjdJVXAzQ1dfLUNVTm1PcmVkdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3OTAxOTY1N30.GnYkuGJjLyxu7lEWqrL9BkxMwLx-7tIM_n63DyLvR9Q'

    extract_neuprint_data(option, token, neuprint_data_location)

# save degree data in variables
in_degree_elements, out_degree_elements = get_neuprint_data(option, neuprint_data_location)

parameters = get_optimised_parameters(optimised_parameters_location)

# could get and change parameters e.g.:
# parameters["N_total_nodes"] = 1000
# parameters["dimensions"] = [2, 0, 2, 0, 2, 0]

# or could get seed used for optimisation:
rng = np.random.default_rng(parameters["int_random_generator"])

graph = optimised_convolutive_graph(rng, parameters, in_degree_elements, out_degree_elements)

print(graph)