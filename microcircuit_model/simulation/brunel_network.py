import shutil
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
from simulation_utils import *
from microcircuit_model.connectivity.connectivity_utils import *
import time
import json

# def injected_dc_current(time_value, lower_limit, upper_limit, current_value):
#     in_time_interval = np.logical_and(lower_limit <= time_value, upper_limit >= time_value)
#     return current_value*in_time_interval
start_program = time.time()
parameter_filename = "config.json"

(int_for_random_generator, number_of_cells, ratio_excitatory_cells, gamma_ratio_connections,
 epsilon_connection_probability, EL, V_reset, Rm, tau_E, tau_I, V_th, refractory_period, transmission_delay,
 J_PSP_amplitude_excitatory, ratio_external_freq_to_threshold_freq, g_inh, simulation_time, time_step,
 save_voltage_data_every_ms, number_of_progression_updates, number_of_scatter_plot_cells,
 convolutive_connectivity, json_parameters) = get_brunel_parameters(parameter_filename)

system_arguments = sys.argv
g_inh_argument_name = "g_inh"
ratio_freq_argument_name = "ratio_freq"
rng_int_argument_name = "rng_int"
if g_inh_argument_name in system_arguments:
    g_inh_argument_index = system_arguments.index(g_inh_argument_name)
    g_inh = float(system_arguments[g_inh_argument_index + 1])
    json_parameters['g_inh'] = g_inh
if ratio_freq_argument_name in system_arguments:
    ratio_freq_argument_index = system_arguments.index(ratio_freq_argument_name)
    ratio_external_freq_to_threshold_freq = float(system_arguments[ratio_freq_argument_index + 1])
    json_parameters['ratio_external_freq_to_threshold_freq'] = ratio_external_freq_to_threshold_freq
if rng_int_argument_name in system_arguments:
    rng_int_argument_index = system_arguments.index(rng_int_argument_name)
    int_for_random_generator = int(system_arguments[rng_int_argument_index + 1])
    json_parameters['int_for_random_generator'] = int_for_random_generator

# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
# option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"
relative_path_to_connectivity = "../connectivity/"

end_parameters = time.time()
print("get parameters: "+str(end_parameters-start_program)+" s")

if not os.path.exists('saved_data_brunel_network'):
    os.makedirs('saved_data_brunel_network')

folder_identifier = 0
while os.path.exists("saved_data_brunel_network/" + str(folder_identifier)):
    folder_identifier += 1
str_identifier = "saved_data_brunel_network/" + str(folder_identifier)
os.makedirs(str_identifier)

with open(str_identifier + "/config.json", 'w') as new_config_file:
    json.dump(json_parameters, new_config_file)

end_folders_save_parameters = time.time()
print("create folder and save parameters: "+str(end_folders_save_parameters-end_parameters)+" s")

rng = np.random.default_rng(int_for_random_generator)
number_excitatory_cells = round(ratio_excitatory_cells * number_of_cells)
number_inhibitory_cells = number_of_cells - number_excitatory_cells
C_random_excitatory_connections = round(number_excitatory_cells * epsilon_connection_probability)
C_random_inhibitory_connections = round(gamma_ratio_connections * C_random_excitatory_connections)
C_random_connections = C_random_excitatory_connections + C_random_inhibitory_connections
C_ext_connections = C_random_excitatory_connections

if convolutive_connectivity:
    connectivity_matrix_ordered = get_optimised_connectivity(option, N_total_nodes=number_of_cells,
                                                             relative_path_to_connectivity=relative_path_to_connectivity,
                                                             int_random_generator=int_for_random_generator)
    reorder_nodes_connectivity = rng.permutation(number_of_cells)
    connectivity_matrix = connectivity_matrix_ordered[reorder_nodes_connectivity, :][:, reorder_nodes_connectivity]
else:
    connectivity_matrix = constant_probability_random_connectivity_matrix(number_of_cells,
                                                                          epsilon_connection_probability, rng)


# the receiving (postsynaptic) cell corresponds to the row,
# and the (presynaptic) cells which give it a connection correspond to columns in that row which have a value
# of 1. So if it is [[0, 1], [0, 0]] then the first cell (index 0) has a connection
# which it receives from cell with index 1.

tau_vector = np.block([np.ones(number_excitatory_cells)*tau_E, np.ones([number_inhibitory_cells])*tau_I])

J_PSP_amplitude_inhibitory = J_PSP_amplitude_excitatory  # mV (PSP is postsynaptic potential amplitude)
freq_threshold = V_th/(J_PSP_amplitude_excitatory*C_random_excitatory_connections*tau_E)  # kHz
external_freq_excitatory = ratio_external_freq_to_threshold_freq*freq_threshold  # kHz (is varied for different simulations)
external_freq_inhibitory = external_freq_excitatory
# synapse properties
# excitatory
g_exc = g_inh  # unitless (so not conductances here)

# noinspection PyTypeChecker
J_PSP_amplitude_matrix = np.block([
    [np.ones((number_excitatory_cells, number_excitatory_cells))*J_PSP_amplitude_excitatory,  # exc --> exc
     np.ones((number_excitatory_cells, number_inhibitory_cells))*-g_exc*J_PSP_amplitude_excitatory],  # inh --> exc
    [np.ones((number_inhibitory_cells, number_excitatory_cells))*J_PSP_amplitude_inhibitory,  # exc --> inh
     np.ones((number_inhibitory_cells, number_inhibitory_cells))*-g_inh*J_PSP_amplitude_inhibitory]  # inh --> inh
])
weighted_connectivity_matrix = connectivity_matrix * J_PSP_amplitude_matrix

transmission_delay_steps = round(transmission_delay/time_step)

spike_skip_steps = round(refractory_period/time_step)
if spike_skip_steps == 0:
    spike_skip_steps = 1
number_of_time_steps = round(simulation_time/time_step)
time_array = np.arange(0, simulation_time, time_step)
V_t = np.zeros((number_of_cells, 2))  # voltage at each time
V_t[:, 0] = EL  # the voltage for the cells has a potential at t=0 equal to EL
save_voltage_data_every_step = round(save_voltage_data_every_ms/time_step)  # step
saved_voltage_data = np.zeros((number_of_cells, round(simulation_time/save_voltage_data_every_ms)))
I_t = np.zeros(number_of_cells)  # if Rm is in M ohm, then I_t is in nA. (Current at each time)
cells_in_refractory_period = np.zeros(number_of_cells) - 1

start_spike_generation = time.time()
print("setup simulation: "+str(start_spike_generation-end_folders_save_parameters)+" s")
external_generated_spike_times = binomial_probability_spike_generation(external_freq_excitatory,
                                                                          time_step, number_of_time_steps,
                                                                          number_of_cells, C_ext_connections, rng)

end_spike_generation = time.time()
print("External spike generation: "+str(end_spike_generation - start_spike_generation)+" s")

k = 0  # index for the timestep.
spikes = {}  # emtpy dict for the spikes with the key the time step (not time value) and the value the spikes cell(s)
number_of_progression_updates = 10
print_progression_every = round(number_of_time_steps/number_of_progression_updates)
print("Start simulation:")
start_simulation = time.time()
while k < number_of_time_steps-1:
    current_index = k % 2
    next_index = (k + 1) % 2
    # cells with cell dynamics which also spiked (therefore have no dynamics next timestep)
    cells_spiked = np.logical_and(np.invert(np.isnan(V_t[:, current_index])), (V_t[:, current_index] >= V_th))
    indices_cells_spiked = np.where(cells_spiked)[0]
    if k % print_progression_every == 0:
        progress = k // print_progression_every
        print(str(progress*100/number_of_progression_updates)+"%")

    if indices_cells_spiked.size != 0:
        spikes[k] = indices_cells_spiked

    # for the cells that spiked, start the refractory period
    cells_in_refractory_period[cells_spiked] = spike_skip_steps
    # for the cells which have refractory period left, set the V_t to np.NaN
    V_t[cells_in_refractory_period >= 1, current_index:] = np.NaN
    # for the cells which have finished refractory period, set the voltage to its reset voltage
    V_t[cells_in_refractory_period == 0] = V_reset

    # determine injected current
    spiked_external_synapses_each_cell = external_generated_spike_times[:, k]
    I_t[:] = tau_vector/Rm*J_PSP_amplitude_excitatory*spiked_external_synapses_each_cell
    # cells with cell dynamics (from this timestep)
    cells_dynamics = np.invert(np.isnan(V_t[:, current_index]))
    cells_dynamics_matrix = np.logical_and(cells_dynamics[:, None], cells_dynamics[None, :])
    num_cells_dynamics = np.sum(cells_dynamics)
    dynamics_shape = (num_cells_dynamics, num_cells_dynamics)
    # spiked_connected_result = np.zeros(weighted_connectivity_matrix.shape)
    spiked_connected_result = np.zeros(number_of_cells)
    if k-transmission_delay_steps in spikes:
        if spikes.get(k-transmission_delay_steps).size != 0:
            spikes_cells_boolean = np.zeros(number_of_cells, dtype=bool)
            spikes_cells_boolean[spikes.get(k-transmission_delay_steps)] = True
            # print("arriving spikes: " + str(spikes.get(k-transmission_delay_steps)))
            # print("spikes_cells_boolean: " + str(spikes_cells_boolean[spikes.get(k-transmission_delay_steps)]))
            # spikes
            # print("total spikes_cells_boolean: " + str(np.sum(spikes_cells_boolean)))
            spiked_connected_result[cells_dynamics] = \
                weighted_connectivity_matrix[cells_dynamics, :] @ spikes_cells_boolean
            # spiked_connected_result[cells_dynamics] = \
            #     weighted_connectivity_matrix[cells_dynamics_matrix].reshape(dynamics_shape) @ spikes_cells_boolean[cells_dynamics]
            # print("1 total spiked_connected_result: " + str(np.sum(spiked_connected_result)))
            # print("spiked_connected_result: " + str(spiked_connected_result[spikes.get(k-transmission_delay_steps)]))
            # ^ gives synaptic weight if both a spike has occurred in the pre-synaptic cell and
            # a connection is present between the pre-synaptic cell and the postsynaptic cell

    # add synaptic currents to injected current to get total current going into the particular cells.
    # print("total I_t before: " + str(np.sum(I_t)))
    total_before = np.sum(I_t)
    I_t[cells_dynamics] += tau_vector[cells_dynamics] / Rm * \
                              spiked_connected_result[cells_dynamics]
    # print("total I_t after: " + str(np.sum(I_t)))
    # print("total I_t difference: " + str(np.sum(I_t)-total_before))
    # standard dynamics, when a spike has NOT recently occurred. (Implicit/Backward Euler)
    V_t[cells_dynamics, next_index] = (tau_vector[cells_dynamics] * V_t[cells_dynamics, current_index] +
                                   time_step * EL + Rm * I_t[cells_dynamics]) / (tau_vector[cells_dynamics] + time_step)
    # Forward Euler
    # V_t[cells_dynamics, k + 1] = (1 - time_step / tau_vector[cells_dynamics]) * V_t[cells_dynamics, k] + \
    #                              time_step/tau_vector[cells_dynamics] * (Rm * I_t[cells_dynamics, k])
    if k % save_voltage_data_every_step == 0:
        saved_voltage_data[:, k // save_voltage_data_every_step] = np.copy(V_t[:, current_index])
    cells_in_refractory_period[cells_in_refractory_period >= 0] -= 1
    k += 1

end_simulation = time.time()
print("Simulation: "+str(end_simulation - start_simulation)+" s")
# print(np.nanmean(V_t[0:5, 50:]))
print("End simulation, start saving data")

with open(str_identifier + "/spikes.npy", "wb") as spikesfile:
    np.save(spikesfile, spikes)

with open(str_identifier + "/voltage_traces.npy", "wb") as voltage_file:
    np.save(voltage_file, saved_voltage_data)

with open(str_identifier + "/binomial_external_spikes.npy", "wb") as external_spikes_file:
    np.save(external_spikes_file, external_generated_spike_times)

end_saving_data = time.time()
print("Saving data: "+str(end_saving_data-end_simulation)+" s")
