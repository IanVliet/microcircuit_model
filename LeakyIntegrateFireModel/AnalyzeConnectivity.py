import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
from FunctionsExpandedBrunelNetwork import *

rng = np.random.default_rng(42)
# web application: https://gtalg.ebrains-italy.eu/connect/
sigma_random_variable_choice = 10  # not present in web application? (Noise factor?, Delta?)
a_arbitrary_constant_choice = 2  # unused / has no influence
m_0_nodes_choice = 100  # equivalent to "m" in the web application?
rho_probability_choice = 0.3  # not present in web application? (Noise factor?, Delta?)
l_cardinality_partitions_choice = 3  # Likely L in the web application
N_total_nodes_choice = 1500  # likely N in the web application
E_k_choice = 3  # likely EK in the web application
phi_U_probability_choice = 0.9  # likely phi_U in the web application
phi_D_probability_choice = 0  # likely phi_D in the web application
p_probability_choice = (E_k_choice/N_total_nodes_choice - phi_D_probability_choice) /\
                       (phi_U_probability_choice - phi_D_probability_choice)

start_generation_B = time.time()
B = convolutive_graph_gen(sigma_random_variable_choice, a_arbitrary_constant_choice, m_0_nodes_choice,
                          rho_probability_choice, l_cardinality_partitions_choice, N_total_nodes_choice,
                          p_probability_choice, phi_U_probability_choice, phi_D_probability_choice, rng)
end_generation_B = time.time()
print(str(end_generation_B-start_generation_B) + " s")

fig_indegrees, ax_indegrees = plt.subplots()
ax_indegrees.hist(np.sum(B, axis=1), bins=25)
ax_indegrees.set_xlabel("indegree")
ax_indegrees.set_ylabel("count")

fig_outdegrees, ax_outdegrees = plt.subplots()
ax_outdegrees.hist(np.sum(B, axis=0), bins=20)
ax_outdegrees.set_xlabel("outdegree")
ax_outdegrees.set_ylabel("count")

plt.show()
