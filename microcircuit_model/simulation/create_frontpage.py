# import matlab.engine
import os
import sys
import numpy as np

from microcircuit_model.connectivity.connectivity_utils import get_optimised_connectivity

sys.path.append(os.path.dirname(sys.path[0]))
from simulation_utils import *

# import scipy

# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
# option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"
# option = "pyc-pyc"
# from MICrONS Explorer https://www.microns-explorer.org/phase1, raw data at: https://zenodo.org/records/5579388
# accompanying paper: Turner, N. L., Macrina, T., Bae, J. A., Yang, R., Wilson, A. M., Schneider-Mizell, C., Lee, K.,
# Lu, R., Wu, J., Bodor, A. L., Bleckert, A. A., Brittain, D., Froudarakis, E., Dorkenwald, S., Collman, F., Kemnitz,
# N., Ih, D., Silversmith, W. M., Zung, J., … Seung, H. S. (2022). Reconstruction of neocortex: Organelles,
# compartments, cells, circuits, and activity. Cell, 185(6), 1082-1100.e24. https://doi.org/10.1016/j.cell.2022.01.023
relative_path_to_connectivity = "../connectivity/"

graph, positions = get_optimised_connectivity(option, return_positions=True, relative_path_to_connectivity=relative_path_to_connectivity)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

log_type = True
config_name = "/config.json"
spikes_name = "/spikes.npy"

identifier = str(43)

str_identifier = "saved_data_brunel_network/" + identifier

parameter_filename = str_identifier + config_name

(int_for_random_generator, number_of_cells, ratio_excitatory_cells, gamma_ratio_connections,
 epsilon_connection_probability, EL, V_reset, Rm, tau_E, tau_I, V_th, refractory_period, transmission_delay,
 J_PSP_amplitude_excitatory, ratio_external_freq_to_threshold_freq, g_inh, simulation_time, time_step,
 save_voltage_data_every_ms, number_of_progression_updates, number_of_scatter_plot_cells,
 convolutive_connectivity, json_parameters) = get_brunel_parameters(parameter_filename)

number_of_time_steps = round(simulation_time/time_step)
rng = np.random.default_rng(int_for_random_generator)
time_array = np.arange(0, simulation_time, time_step)
save_voltage_data_every_step = round(save_voltage_data_every_ms/time_step)  # step
number_excitatory_cells = round(ratio_excitatory_cells * number_of_cells)
C_random_excitatory_connections = round(number_excitatory_cells * epsilon_connection_probability)

type_of_experimental_system = ("computational brunel network with g=" + str(g_inh) + " and v_ext/v_thr=" +
                               str(ratio_external_freq_to_threshold_freq))
png_extension = ".png"
pdf_extension = ".pdf"

if os.path.exists(str_identifier + spikes_name):
    with open(str_identifier + spikes_name, 'rb') as spikes_file:
        spikes = np.load(spikes_file, allow_pickle=True)
        spikes = spikes[()]
    spikes_binary_array = np.zeros((number_of_cells, number_of_time_steps), dtype=bool)
    for spike_time, spiked_cells in spikes.items():
        spikes_binary_array[spikes.get(spike_time), spike_time] = True

    summed_spikes = np.sum(spikes_binary_array[:, :], axis=1)
    # summed_spikes = np.ones(number_of_cells)

    for position, spike_value in zip(positions[0:2, :].T, summed_spikes):
        x_idx = np.argmin(np.abs(x - position[0]))
        y_idx = np.argmin(np.abs(y - position[1]))
        Z[y_idx, x_idx] = spike_value

    # Plot the image
    fig, ax = plt.subplots()
    ax.imshow(Z, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    # ax.colorbar(label='Values')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Image with positions and values')
    plt.show()
