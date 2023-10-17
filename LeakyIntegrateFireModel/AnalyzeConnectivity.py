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
l_cardinality_partitions_choice = 3  # Likely L in the web application
N_total_nodes_choice = 15000  # likely N in the web application
E_k_choice = 3  # likely EK in the web application
phi_U_probability_choice = 0.9  # likely phi_U in the web application
phi_D_probability_choice = 0  # likely phi_D in the web application
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
plt.show()
