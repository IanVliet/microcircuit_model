import shutil
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from simulation_utils import *
from matplotlib.ticker import PercentFormatter
import time
import json

folders = range(67, 84, 4)
left_lim, right_lim = 500, 600

main_folder = "saved_data_brunel_network/"
spikes_name = "/spikes.npy"
config_name = "/config.json"

firing_rates = np.zeros(len(folders))
global_oscillation_frequencies = np.zeros(len(folders))
total_spikes_arrays = []
time_arrays = []  # it will be the latest time array acquired from the for loop

for index, folder in enumerate(folders):
    # newer version where both average firing rate and global oscillation frequency were saved
    if os.path.isfile(main_folder + str(folder) + "/firing_statistics.json"):
        with open(main_folder + str(folder) + "/firing_statistics.json", "r") as firing_statistics_file:
            avg_firing_rate_dict = json.load(firing_statistics_file)
    # old version where only average firing rate was saved
    elif os.path.isfile(main_folder + str(folder) + "/avg_firing_rate.json"):
        with open(main_folder + str(folder) + "/avg_firing_rate.json", "r") as avg_firing_rate_file:
            avg_firing_rate_dict = json.load(avg_firing_rate_file)

    firing_rates[index] = avg_firing_rate_dict["avg_firing_rate"]

    (int_for_random_generator, number_of_cells, ratio_excitatory_cells, gamma_ratio_connections,
     epsilon_connection_probability, EL, V_reset, Rm, tau_E, tau_I, V_th, refractory_period, transmission_delay,
     J_PSP_amplitude_excitatory, ratio_external_freq_to_threshold_freq, g_inh, simulation_time, time_step,
     save_voltage_data_every_ms, number_of_progression_updates, number_of_scatter_plot_cells,
     convolutive_connectivity, json_parameters) = get_brunel_parameters(main_folder + str(folder) + config_name)

    number_of_time_steps = round(simulation_time / time_step)
    rng = np.random.default_rng(int_for_random_generator)
    time_array = np.arange(0, simulation_time, time_step)
    time_arrays.append(time_array)
    save_voltage_data_every_step = round(save_voltage_data_every_ms / time_step)  # step
    number_excitatory_cells = round(ratio_excitatory_cells * number_of_cells)
    C_random_excitatory_connections = round(number_excitatory_cells * epsilon_connection_probability)

    with open(main_folder + str(folder) + spikes_name, 'rb') as spikes_file:
        spikes = np.load(spikes_file, allow_pickle=True)
        spikes = spikes[()]

    total_spikes = np.zeros(number_of_time_steps)
    for spike_time, spiked_cells in spikes.items():
        total_spikes[spike_time] += len(spiked_cells)

    total_spikes_arrays.append(total_spikes)

    total_spikes_fft = np.fft.fft(total_spikes)
    total_spikes_power_spec = np.abs(total_spikes_fft) ** 2
    frequencies = np.fft.fftfreq(time_array.shape[-1], d=time_step)
    global_oscillation_frequencies[index] = 1e3 * frequencies[1:len(total_spikes_power_spec) // 2][np.argmax(total_spikes_power_spec[1:len(total_spikes_power_spec) // 2])]

# aligned_total_spikes_arrays, time_lags = align_arrays(total_spikes_arrays)
# print(time_lags)
# fig_instant_freq, ax_instant_freq = plt.subplots(2)
# for count, (time_lag, aligned_total_spikes, total_spikes_plot) in enumerate(zip(time_lags, aligned_total_spikes_arrays, total_spikes_arrays)):
#     aligned_time_array = shift_array(time_arrays[count], time_lag)
#     ax_instant_freq[0].bar(aligned_time_array, aligned_total_spikes, label=str(count))
#
#     ax_instant_freq[1].bar(time_arrays[count], total_spikes_plot, label=str(count))
# # ax_instant_freq.set_xlim(left_lim, right_lim)
# ax_instant_freq[0].set_xlabel('time (ms)')
# ax_instant_freq[0].set_ylabel(r'total number of spikes')
# ax_instant_freq[0].legend()
# ax_instant_freq[1].set_xlabel('time (ms)')
# ax_instant_freq[1].set_ylabel(r'total number of spikes')
# ax_instant_freq[1].legend()

mean_avg_firing_rate = np.mean(firing_rates)
std_avg_firing_rate = np.std(firing_rates, ddof=1)
print(mean_avg_firing_rate)
print(std_avg_firing_rate)

mean_global_oscillation_frequencies = np.mean(global_oscillation_frequencies)
std_global_oscillation_frequencies = np.std(global_oscillation_frequencies, ddof=1)
print(mean_global_oscillation_frequencies)
print(std_global_oscillation_frequencies)

if not os.path.exists('saved_combined_data_brunel_network'):
    os.makedirs('saved_combined_data_brunel_network')

folder_identifier = 0
while os.path.exists("saved_combined_data_brunel_network/" + str(folder_identifier)):
    folder_identifier += 1
str_identifier = "saved_combined_data_brunel_network/" + str(folder_identifier)
os.makedirs(str_identifier)

combined_activity_dict = {
    "mean_avg_firing_rate": mean_avg_firing_rate,
    "std_avg_firing_rate": std_avg_firing_rate,
    "mean_global_oscillation_frequencies": mean_global_oscillation_frequencies,
    "std_global_oscillation_frequencies": std_global_oscillation_frequencies,
    "folders": list(folders)
                          }
with open(str_identifier + "/combined_activity.json", "w") as combined_activity_file:
    json.dump(combined_activity_dict, combined_activity_file)

plt.show()
