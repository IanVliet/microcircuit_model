import matplotlib.pyplot as plt
import numpy as np

from connectivity_utils import *

# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
# option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"
# option = "pyc-pyc"
# from MICrONS Explorer https://www.microns-explorer.org/phase1, raw data at: https://zenodo.org/records/5579388
# accompanying paper: Turner, N. L., Macrina, T., Bae, J. A., Yang, R., Wilson, A. M., Schneider-Mizell, C., Lee, K.,
# Lu, R., Wu, J., Bodor, A. L., Bleckert, A. A., Brittain, D., Froudarakis, E., Dorkenwald, S., Collman, F., Kemnitz,
# N., Ih, D., Silversmith, W. M., Zung, J., … Seung, H. S. (2022). Reconstruction of neocortex: Organelles,
# compartments, cells, circuits, and activity. Cell, 185(6), 1082-1100.e24. https://doi.org/10.1016/j.cell.2022.01.023

graph = get_optimised_connectivity(option)
number_of_nodes = len(graph)

average_probability_sim = np.sum(graph)/((number_of_nodes-1)*number_of_nodes)
print(average_probability_sim)
print("J_new = J_old * eps_old / eps_new = ", 0.1*0.1/average_probability_sim)

in_degree_elements, out_degree_elements = get_degree_data(option)
average_probability_data_indegree = np.average(in_degree_elements)/(number_of_nodes-1)
average_probability_data_outdegree = np.average(out_degree_elements)/(number_of_nodes-1)
print(average_probability_data_indegree, average_probability_data_outdegree)


print(graph)
fig, ax = plt.subplots()
ax.imshow(graph[:, np.argsort(graph.sum(axis=0))][np.argsort(graph.sum(axis=1)), :])

plt.show()
