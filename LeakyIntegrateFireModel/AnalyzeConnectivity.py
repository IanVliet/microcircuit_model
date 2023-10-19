import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
from FunctionsExpandedBrunelNetwork import *

in_degree_elements, out_degree_elements = \
    get_csv_degree_elements("connectome layer 4 of the somatosensory cortex indegree.csv",
                                 "connectome layer 4 of the somatosensory cortex outdegree.csv")

rng = np.random.default_rng(42)
# web application: https://gtalg.ebrains-italy.eu/connect/
# sigma_random_variable_choice = 10  # not present in web application? (Noise factor?, Delta?)
# a_arbitrary_constant_choice = 2  # unused / has no influence
m_0_nodes_choice = 100  # equivalent to "m" in the web application?
rho_probability_choice = 0.3  # not present in web application? (Noise factor?, Delta?)
l_cardinality_partitions_choice = 1500  # Likely L in the web application
N_total_nodes_choice = 1500  # likely N in the web application
E_k_choice = 3  # likely EK in the web application
phi_U_probability_choice = 0.9  # likely phi_U in the web application
phi_D_probability_choice = 0.0005  # likely phi_D in the web application
p_probability_choice = (E_k_choice/N_total_nodes_choice - phi_D_probability_choice) /\
                       (phi_U_probability_choice - phi_D_probability_choice)

start_generation = time.time()
B = convolutive_graph_gen(m_0_nodes_choice, rho_probability_choice, l_cardinality_partitions_choice,
                          N_total_nodes_choice, p_probability_choice, phi_U_probability_choice,
                          phi_D_probability_choice, rng, spatial=True, in_degree_elements=in_degree_elements)

# A = spatial_block_ba_graph(N_total_nodes_choice, 1, 0, 1, 0, 1, 0, 1.5, 3, 0, rng,
#                            in_degree_elements, out_degree_elements)[0]
end_generation = time.time()
print(str(end_generation-start_generation) + " s")
plot_degree_counts(B)

in_degree, out_degree = \
    get_interpolated_degree_distributions(in_degree_elements, out_degree_elements)

in_degrees, in_degree_counts = np.unique(np.sum(B, axis=1), return_counts=True)
out_degrees, out_degree_counts = np.unique(np.sum(B, axis=0), return_counts=True)
in_degree_elements_model = np.repeat(in_degrees, in_degree_counts)
out_degree_elements_model = np.repeat(out_degrees, out_degree_counts)

in_degree_gen_model, out_degree_gen_model = \
    get_interpolated_degree_distributions(in_degree_elements_model, out_degree_elements_model)

# in_degree_model, out_degree_model = \
#     convolutive_probabilities(in_degree, out_degree, N_total_nodes_choice, l_cardinality_partitions_choice,
#                               p_probability_choice, phi_U_probability_choice, phi_D_probability_choice)

fig, ax = plt.subplots()
ax.plot(in_degree, label="in-degree")
ax.plot(out_degree, label="out-degree")
ax.plot(in_degree_gen_model, label="in-degree generative model")
ax.plot(out_degree_gen_model, label="out-degree generative model")
# ax.plot(in_degree_model, label="in-degree model")
# ax.plot(out_degree_model, label="out-degree model")
ax.legend()

plt.show()
