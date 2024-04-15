import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
import csv
from collections import Counter
import networkx as nx
from scipy.special import comb
from scipy.stats import binom
from scipy.signal import correlate
from scipy.signal import correlation_lags
import json
import sys


def poisson_distribution_spike_generation(freq, total_time, time_step_size, cells, connections, rng):
    # generate external spike trains
    average_number_of_spikes = ceil(freq * total_time)  # freq * time interval ([kHz]*[ms])
    average_interval_in_steps = 1/(freq * time_step_size)
    external_spike_intervals = rng.poisson(average_interval_in_steps,
                                                 size=(cells, connections, average_number_of_spikes))
    # uniform_spike_start = np.random.randint(0, round(average_interval_in_steps*2), (number_of_cells, C_ext_connections))
    # complete_external_spike_intervals = np.dstack((
    #         uniform_spike_start,
    #         external_spike_intervals
    #     ))
    # external_spike_times = np.cumsum(complete_external_spike_intervals, axis=2)
    external_spike_times = np.cumsum(external_spike_intervals, axis=2)
    # minimum_indices = []
    latest_spikes_smallest_value = np.min(external_spike_times[:, :, -1])
    while latest_spikes_smallest_value <= total_time / time_step_size:
        # minimum_indices.append(np.unravel_index(np.argmin(external_spike_times[:, :, -1])
        # , external_spike_times[:, :, -1].shape))
        rest_time = total_time - latest_spikes_smallest_value * time_step_size
        rest_expected_spikes = ceil(freq * rest_time)
        rest_external_spike_intervals = np.dstack((
            external_spike_times[:, :, -1],
            rng.poisson(average_interval_in_steps, size=(cells, connections, rest_expected_spikes))
        ))
        rest_external_spike_times = np.cumsum(rest_external_spike_intervals, axis=2)
        external_spike_times = np.dstack([
            external_spike_times[:, :, :-1], rest_external_spike_times
        ])
        latest_spikes_smallest_value = np.min(external_spike_times[:, :, -1])
    # for indices in minimum_indices:
        # print(external_spike_times[indices])
    return external_spike_times


def uniform_log_spike_generation(freq, total_time, time_step_size, total_time_steps, cells, connections, rng):
    # generate external spike trains --> from "SIMULATING THE POISSON PROCESS" by PATRICK MCQUIGHAN
    average_number_of_spikes = ceil(freq * total_time)  # freq * time interval ([kHz]*[ms])
    uniform_for_external_spike_intervals = \
        rng.uniform(0, 1, (cells, connections, average_number_of_spikes))
    external_spike_intervals = -np.log(1 - uniform_for_external_spike_intervals) / (freq*time_step_size)
    external_spike_times = np.cumsum(external_spike_intervals, axis=2, dtype=int)
    # minimum_indices = []
    while np.min(external_spike_times[:, :, -1]) <= total_time_steps:
        # minimum_indices.append(np.unravel_index(np.argmin(external_spike_times[:, :, -1]),
        #                                         external_spike_times[:, :, -1].shape))
        rest_time = total_time - np.min(external_spike_times[:, :, -1]) * time_step_size
        rest_expected_spikes = ceil(freq * rest_time)
        rest_uniform_for_external_spike_intervals = \
            rng.uniform(0, 1, (cells, connections, rest_expected_spikes))
        rest_external_spike_intervals = np.dstack((
            external_spike_times[:, :, -1],
            -np.log(1 - rest_uniform_for_external_spike_intervals) / (freq*time_step_size)
        ))
        rest_external_spike_times = np.cumsum(rest_external_spike_intervals, axis=2)
        external_spike_times = np.block([
            external_spike_times[:, :, :-1], rest_external_spike_times
        ])
    # for indices in minimum_indices:
    #     print(external_spike_times[indices])
    external_spike_times = np.rint(external_spike_times)
    return external_spike_times


def uniform_probability_spike_generation(freq, total_time, time_step_size, total_time_steps, cells, connections, rng):
    # generate external spike trains --> from "Poisson Model of Spike Generation" by Professor David Heeger
    uniform_for_external_spike_intervals = \
        rng.uniform(0, 1, (cells, connections, total_time_steps))
    external_spike_steps = uniform_for_external_spike_intervals <= freq*time_step_size
    return external_spike_steps


def binomial_probability_spike_generation(freq, time_step_size, total_time_steps, cells, connections, rng):
    # generate external spike trains --> from "Poisson Model of Spike Generation" by Professor David Heeger
    external_spike_steps = np.zeros((cells, total_time_steps), dtype=np.short)
    external_spike_steps[:, :] = \
        rng.binomial(connections, freq*time_step_size, (cells, total_time_steps))
    return external_spike_steps


def get_brunel_parameters(parameter_filename):
    try:
        with open(parameter_filename, "r") as configfile:
            json_parameters = json.load(configfile)

        int_for_random_generator = json_parameters['int_for_random_generator']

        # network properties
        number_of_cells = json_parameters['number_of_cells']
        ratio_excitatory_cells = json_parameters['ratio_excitatory_cells']
        gamma_ratio_connections = json_parameters['gamma_ratio_connections']
        epsilon_connection_probability = json_parameters['epsilon_connection_probability']
        if 'convolutive_connectivity' in json_parameters:
            convolutive_connectivity = json_parameters['convolutive_connectivity']
        else:
            convolutive_connectivity = False
        # epsilon_connection_probability = C_random_excitatory_connections/number_excitatory_cells

        # cell properties https://link.springer.com/article/10.1023/A:1008925309027
        EL = json_parameters['EL']  # mV
        V_reset = json_parameters['V_reset']  # mV
        Rm = json_parameters['Rm']  # M ohm (not needed anymore since RI is defined in completeness
        tau_E = json_parameters['tau_E']  # ms (time constant RC of excitatory neuron)
        tau_I = json_parameters['tau_I']  # ms (time constant RC of inhibitory neuron)
        V_th = json_parameters['V_th']  # mV (threshold voltage (=θ))
        refractory_period = json_parameters['refractory_period']  # ms
        transmission_delay = json_parameters['transmission_delay']  # ms

        # synaptic properties, and external frequency
        J_PSP_amplitude_excitatory = json_parameters[
            'J_PSP_amplitude_excitatory']  # mV (PSP is postsynaptic potential amplitude)
        ratio_external_freq_to_threshold_freq = json_parameters['ratio_external_freq_to_threshold_freq']
        g_inh = json_parameters['g_inh']  # unitless (relative strength of inhibitory connections)

        # simulation properties
        simulation_time = json_parameters['simulation_time']  # ms
        time_step = json_parameters['time_step']  # ms

        save_voltage_data_every_ms = json_parameters[
            'save_voltage_data_every_ms']  # ms  (the time between datapoints for the voltage data)
        number_of_progression_updates = json_parameters[
            'number_of_progression_updates']  # the number of updates given (e.g. if it is 10, it will give 0%, 10%, ... , 90%)
        number_of_scatter_plot_cells = json_parameters['number_of_scatter_plot_cells']

    except IOError:
        print("Could not process the file due to IOError")
        sys.exit(1)
    except KeyError:
        print("A key was not defined")
        sys.exit(2)
    return (int_for_random_generator, number_of_cells, ratio_excitatory_cells, gamma_ratio_connections,
            epsilon_connection_probability, EL, V_reset, Rm, tau_E, tau_I, V_th, refractory_period, transmission_delay,
            J_PSP_amplitude_excitatory, ratio_external_freq_to_threshold_freq, g_inh, simulation_time, time_step,
            save_voltage_data_every_ms, number_of_progression_updates, number_of_scatter_plot_cells,
            convolutive_connectivity, json_parameters)


def align_arrays(arrays):
    if len(arrays) <= 0:
        print("No arrays were given")
        sys.exit(1)
    elif len(arrays) == 1:
        print("Only a single array was given")
        return arrays
    else:
        ref_array = arrays[0]
        aligned_arrays = [ref_array]
        time_lags = np.zeros(len(arrays), dtype=int)
        for index, array in enumerate(arrays[1:]):
            corr = correlate(ref_array, array, mode='full')
            fig, ax = plt.subplots()
            ax.bar(np.arange(len(corr)), corr)
            lags = correlation_lags(ref_array.size, array.size, mode="full")
            lag = lags[np.argmax(corr)]
            time_lags[index + 1] = lag
            aligned_arrays.append(shift_array(array, lag))
        return aligned_arrays, time_lags


def shift_array(array, shift, fill_value=np.NaN):
    result = np.empty_like(array)
    if shift > 0:
        result[:shift] = fill_value
        result[shift:] = array[:-shift]
    elif shift < 0:
        result[shift:] = fill_value
        result[:shift] = array[-shift:]
    else:
        result[:] = array
    return result
