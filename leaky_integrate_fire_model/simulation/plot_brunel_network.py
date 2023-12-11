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

str_identifier = "saved_data_brunel_network/12"
str_identifier_figures = str_identifier + "/figures"
if not os.path.exists(str_identifier_figures):
    os.makedirs(str_identifier_figures)

spikes_name = "/spikes.npy"
voltage_traces_name = "/voltage_traces.npy"
external_spikes_name = "/external_spikes.npy"
alt_external_spikes_name = "/alt_external_spikes.npy"
binomial_external_spikes_name = "/binomial_external_spikes.npy"
config_name = "/config.json"
parameter_filename = str_identifier + config_name

int_for_random_generator, number_of_cells, ratio_excitatory_cells, gamma_ratio_connections, \
        epsilon_connection_probability, EL, V_reset, Rm, tau_E, tau_I, V_th, refractory_period, transmission_delay, \
        J_PSP_amplitude_excitatory, ratio_external_freq_to_threshold_freq, g_inh, simulation_time, time_step, \
        save_voltage_data_every_ms, number_of_progression_updates, number_of_scatter_plot_cells = get_brunel_parameters(parameter_filename)

number_of_time_steps = round(simulation_time/time_step)
rng = np.random.default_rng(int_for_random_generator)
time_array = np.arange(0, simulation_time, time_step)
save_voltage_data_every_step = round(save_voltage_data_every_ms/time_step)  # step
number_excitatory_cells = round(ratio_excitatory_cells * number_of_cells)
C_random_excitatory_connections = round(number_excitatory_cells * epsilon_connection_probability)
png_extension = ".png"
pdf_extension = ".pdf"
left_lim, right_lim = 1000, 1200

if os.path.exists(str_identifier + spikes_name):
    with open(str_identifier + spikes_name, 'rb') as spikes_file:
        spikes = np.load(spikes_file, allow_pickle=True)
        spikes = spikes[()]

    plot_scatter_spikes_name = "/scatter_spikes"
    fig, ax = plt.subplots()
    jet_colors = cm.rainbow(np.linspace(0, 1, number_of_scatter_plot_cells))
    chosen_cells = rng.choice(number_of_cells, number_of_scatter_plot_cells)
    total_spikes = np.zeros(number_of_time_steps)
    for spike_time, spiked_cells in spikes.items():
        total_spikes[spike_time] += len(spiked_cells)
        spiked_chosen_cells = np.intersect1d(spiked_cells, chosen_cells)
        if spiked_chosen_cells.size != 0:
            spiked_chosen_cells_indices = np.where(np.in1d(chosen_cells, spiked_chosen_cells))[0]
            ax.scatter(time_array[spike_time]*np.ones(spiked_chosen_cells_indices.shape),
                       spiked_chosen_cells_indices, color=jet_colors[spiked_chosen_cells_indices])

    ax.set_yticks([])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('cell index')
    ax.set_xlim(left_lim, right_lim)
    fig.savefig(str_identifier_figures + plot_scatter_spikes_name + png_extension)
    fig.savefig(str_identifier_figures + plot_scatter_spikes_name + pdf_extension)

    # instant_freq = total_spikes/time_step
    # total spikes also used in "https://brunomaga.github.io/LIF-Brunel"
    plot_total_spikes_name = "/total_spikes"
    fig_instant_freq, ax_instant_freq = plt.subplots()
    ax_instant_freq.bar(time_array, total_spikes)
    ax_instant_freq.set_xlabel('time (ms)')
    ax_instant_freq.set_ylabel(r'total number of spikes')
    ax_instant_freq.set_xlim(left_lim, right_lim)
    fig_instant_freq.savefig(str_identifier_figures + plot_total_spikes_name + png_extension)
    fig_instant_freq.savefig(str_identifier_figures + plot_total_spikes_name + pdf_extension)

    # get frequency spectrum of total_spikes
    print("Average firing rate: " + str(np.sum(total_spikes)*1e3/simulation_time/number_of_cells) + " Hz")
    plot_freq_spectrum_name = "/freq_spectrum"
    fig_freq_spectrum, ax_freq_spectrum = plt.subplots()
    total_spikes_fft = np.fft.fft(total_spikes)
    total_spikes_power_spec = np.abs(total_spikes_fft)**2
    frequencies = np.fft.fftfreq(time_array.shape[-1], d=time_step)
    ax_freq_spectrum.plot(frequencies, total_spikes_fft.real, label='real')
    ax_freq_spectrum.plot(frequencies, total_spikes_fft.imag, label='imag')
    ax_freq_spectrum.plot(frequencies, total_spikes_power_spec, label='power')
    ax_freq_spectrum.set_xlabel('frequency (kHz)')
    ax_freq_spectrum.set_ylabel(r'Power')
    ax_freq_spectrum.legend()
    fig_freq_spectrum.savefig(str_identifier_figures + plot_freq_spectrum_name + png_extension)
    fig_freq_spectrum.savefig(str_identifier_figures + plot_freq_spectrum_name + pdf_extension)

if os.path.exists(str_identifier + voltage_traces_name):
    with open(str_identifier + voltage_traces_name, 'rb') as voltage_traces_file:
        saved_voltage_data = np.load(voltage_traces_file)

    plot_voltage_traces_name = "/voltage_traces"
    number_of_plotted_cells = 5
    colors = cm.rainbow(np.linspace(0, 1, number_of_plotted_cells))
    line_styles = lines.lineStyles.keys()
    fig_volt, ax_volt = plt.subplots()
    for post_cell_index, post_cell_color in zip(range(number_of_plotted_cells), colors):
        ax_volt.plot(time_array[::save_voltage_data_every_step], saved_voltage_data[post_cell_index, :], color=post_cell_color, label=post_cell_index)
    # ax_volt.plot(time_array, V_t[0, :].T, color=colors[0], label=0)
    ax_volt.set_ylim(-5, 35)
    ax_volt.axhline(V_th, color='r', linestyle=':')
    ax_volt.set_xlabel('time (ms)')
    ax_volt.set_ylabel('membrane potential (mV)')
    ax_volt.legend()
    fig_volt.savefig(str_identifier_figures + plot_voltage_traces_name + png_extension)
    fig_volt.savefig(str_identifier_figures + plot_voltage_traces_name + pdf_extension)

# the version where the spikes are created through a function with log.
if os.path.exists(str_identifier + external_spikes_name):
    with open(str_identifier + external_spikes_name, 'rb') as external_spikes_file:
        external_generated_spike_times = np.load(external_spikes_file)

    plot_external_total_spikes_name = "/external_total_spikes"
    fig_ext_spikes, ax_ext_spikes = plt.subplots()
    ext_spike_count_step_values, ext_spike_count_time_counts = np.unique(external_generated_spike_times, return_counts=True)
    ax_ext_spikes.bar(time_step*ext_spike_count_step_values[ext_spike_count_step_values <= number_of_time_steps],
                      ext_spike_count_time_counts[ext_spike_count_step_values <= number_of_time_steps], label="floats")
    # ax_ext_spikes.hist(external_generated_spike_times.flatten(), bins=number_of_time_steps + 1, label="floats",
    #                    histtype='step')
    ax_ext_spikes.set_xlim(0, simulation_time)
    ax_ext_spikes.set_xlabel("spike times (ms)")
    ax_ext_spikes.set_ylabel("total spike count")

    plot_external_spikes_per_connection_name = "/external_spikes_per_connection"
    fig_ext_spikes_count, ax_ext_spikes_count = plt.subplots()
    spike_count_per_connection_ext = np.count_nonzero(external_generated_spike_times <= number_of_time_steps, axis=2)
    # probability_spike_count = spike_count_per_connection_ext/np.sum(spike_count_per_connection_ext)*100
    ax_ext_spikes_count.hist(spike_count_per_connection_ext.flatten(), bins=10, density=True, histtype='step',
                             label="floats")
    ax_ext_spikes_count.set_xlabel("spike count per connection")
    ax_ext_spikes_count.set_ylabel("probability (%)")
    ax_ext_spikes_count.yaxis.set_major_formatter(PercentFormatter(1.0))

    plot_external_spikes_per_cell_name = "/external_spikes_per_cell"
    fig_ext_spikes_count_cell, ax_ext_spikes_count_cell = plt.subplots()
    spike_count_per_cell_ext = np.sum(np.count_nonzero(external_generated_spike_times <= number_of_time_steps, axis=2), axis=1)
    # probability_spike_count = spike_count_per_connection_ext/np.sum(spike_count_per_connection_ext)*100
    ax_ext_spikes_count_cell.hist(spike_count_per_cell_ext.flatten(), bins=10, density=True, histtype='step',
                             label="floats")
    ax_ext_spikes_count_cell.set_xlabel("spike count per cell")
    ax_ext_spikes_count_cell.set_ylabel("probability (%)")
    ax_ext_spikes_count_cell.yaxis.set_major_formatter(PercentFormatter(1.0))

    plot_external_spike_interval_name = "/external_spike_interval"
    fig_ext_spikes_dif, ax_ext_spikes_dif = plt.subplots()
    ax_ext_spikes_dif.hist(np.diff(external_generated_spike_times).flatten(), bins=number_of_time_steps + 1, density=True)
    ax_ext_spikes_dif.set_xlabel("spike interval (ms)")
    ax_ext_spikes_dif.set_ylabel("probability (%)")
    ax_ext_spikes_dif.yaxis.set_major_formatter(PercentFormatter(1.0))

    # the version where the spikes are also created through the binomial function.
    if os.path.exists(str_identifier + alt_external_spikes_name):
        with open(str_identifier + alt_external_spikes_name, 'rb') as alt_external_spikes_file:
            alt_external_generated_spike_times = np.load(alt_external_spikes_file)
        ax_ext_spikes.bar(time_array, np.sum(alt_external_generated_spike_times, axis=0), label="alt", color="red")
        ax_ext_spikes.legend()

        ax_ext_spikes_count_cell.hist(np.sum(alt_external_generated_spike_times, axis=1),
                                 bins=10, density=True, label="average alt", color="red")
        ax_ext_spikes_count_cell.legend()

    fig_ext_spikes.savefig(str_identifier_figures + plot_external_total_spikes_name + png_extension)
    fig_ext_spikes.savefig(str_identifier_figures + plot_external_total_spikes_name + pdf_extension)

    fig_ext_spikes_count.savefig(str_identifier_figures + plot_external_spikes_per_connection_name + png_extension)
    fig_ext_spikes_count.savefig(str_identifier_figures + plot_external_spikes_per_connection_name + pdf_extension)

    fig_ext_spikes_count_cell.savefig(str_identifier_figures + plot_external_spikes_per_cell_name + png_extension)
    fig_ext_spikes_count_cell.savefig(str_identifier_figures + plot_external_spikes_per_cell_name + pdf_extension)

    fig_ext_spikes_dif.savefig(str_identifier_figures + plot_external_spike_interval_name + png_extension)
    fig_ext_spikes_dif.savefig(str_identifier_figures + plot_external_spike_interval_name + pdf_extension)

# the version where the spikes are only created through the binomial function.
if os.path.exists(str_identifier + binomial_external_spikes_name):
    with open(str_identifier + binomial_external_spikes_name, 'rb') as external_spikes_file:
        external_generated_spike_times = np.load(external_spikes_file)

    plot_external_total_spikes_name = "/external_total_spikes"
    fig_ext_spikes, ax_ext_spikes = plt.subplots()
    ax_ext_spikes.bar(time_array, np.sum(external_generated_spike_times, axis=0))
    ax_ext_spikes.set_xlim(0, simulation_time)
    ax_ext_spikes.set_xlabel("spike times (ms)")
    ax_ext_spikes.set_ylabel("total spike count")

    plot_external_spikes_per_cell_name = "/external_spikes_per_cell"
    fig_ext_spikes_count_cell, ax_ext_spikes_count_cell = plt.subplots()
    ax_ext_spikes_count_cell.hist(np.sum(external_generated_spike_times, axis=1),
                                  bins=10, density=True)
    ax_ext_spikes_count_cell.set_xlabel("spike count per cell")
    ax_ext_spikes_count_cell.set_ylabel("probability (%)")
    ax_ext_spikes_count_cell.yaxis.set_major_formatter(PercentFormatter(1.0))

    fig_ext_spikes.savefig(str_identifier_figures + plot_external_total_spikes_name + png_extension)
    fig_ext_spikes.savefig(str_identifier_figures + plot_external_total_spikes_name + pdf_extension)

    fig_ext_spikes_count_cell.savefig(str_identifier_figures + plot_external_spikes_per_cell_name + png_extension)
    fig_ext_spikes_count_cell.savefig(str_identifier_figures + plot_external_spikes_per_cell_name + pdf_extension)


plt.show()
