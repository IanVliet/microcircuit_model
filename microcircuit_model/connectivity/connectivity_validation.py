from connectivity_utils import *

# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
# option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"

in_degree_elements, out_degree_elements = get_degree_data(option)

graph = get_optimised_connectivity(option)


# int_random_number_generator = 2
# rng = np.random.default_rng(int_random_number_generator)
#
# subset_in_degree_elements, subset_out_degree_elements, rest_in_degree_elements, rest_out_degree_elements = (
#     randomly_select_subset_degree_data(in_degree_elements, out_degree_elements, rng))
#
# print(subset_in_degree_elements)
