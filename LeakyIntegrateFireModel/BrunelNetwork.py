import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time

# def injected_dc_current(time_value, lower_limit, upper_limit, current_value):
#     in_time_interval = np.logical_and(lower_limit <= time_value, upper_limit >= time_value)
#     return current_value*in_time_interval


rng = np.random.default_rng(42)


def poisson_distribution_spike_generation(freq, total_time, time_step_size, cells, connections):
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


def uniform_log_spike_generation(freq, total_time, time_step_size, total_time_steps, cells, connections):
    # generate external spike trains
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


def uniform_probability_spike_generation(freq, total_time, time_step_size, total_time_steps, cells, connections):
    # generate external spike trains
    uniform_for_external_spike_intervals = \
        rng.uniform(0, 1, (cells, connections, total_time_steps))
    external_spike_steps = uniform_for_external_spike_intervals <= freq*time_step_size
    return external_spike_steps

start_program = time.time()
number_of_cells = 1250
number_excitatory_cells = round(0.8*number_of_cells)
number_inhibitory_cells = number_of_cells - number_excitatory_cells
gamma_ratio_connections = 0.25
C_random_excitatory_connections = 1000
C_random_inhibitory_connections = round(gamma_ratio_connections * C_random_excitatory_connections)
C_random_connections = C_random_excitatory_connections + C_random_inhibitory_connections
C_ext_connections = C_random_excitatory_connections
epsilon_connection_probability = 0.1  # C_random_excitatory_connections/number_excitatory_cells
connectivity_matrix = np.random.uniform(size=(number_of_cells, number_of_cells)) <= epsilon_connection_probability
# the receiving (postsynaptic) cell corresponds to the row,
# and the (presynaptic) cells which give it a connection correspond to columns in that row which have a value
# of 1. So if it is [[0, 1], [0, 0]] then the first cell (index 0) has a connection
# which it receives from cell with index 1.

# cell properties https://link.springer.com/article/10.1023/A:1008925309027
EL = 0  # mV
V_reset = 10  # mV
Rm = 1  # M ohm (not needed anymore since RI is defined in completeness
tau_E = 20  # ms (time constant RC of excitatory neuron)
tau_I = 20  # ms (time constant RC of inhibitory neuron)
V_th = 20  # mV (threshold voltage (=θ))
refractory_period = 2.0  # ms
transmission_delay = 1.5  # ms

tau_vector = np.block([np.ones(number_excitatory_cells)*tau_E, np.ones([number_inhibitory_cells])*tau_I])

J_PSP_amplitude_excitatory = 0.1  # mV (PSP is postsynaptic potential amplitude)
J_PSP_amplitude_inhibitory = J_PSP_amplitude_excitatory  # mV (PSP is postsynaptic potential amplitude)
freq_threshold = V_th/(J_PSP_amplitude_excitatory*C_random_excitatory_connections*tau_E)  # kHz
external_freq_excitatory = 0.9*freq_threshold  # kHz (is varied for different simulations)
external_freq_inhibitory = external_freq_excitatory
print(external_freq_excitatory)
# synapse properties
# excitatory
g_exc = 1  # unitless (so not conductances here)
# inhibitory
g_inh = 4.5  # unitless (so not conductances here) (relative strength of inhibitory connections)

# noinspection PyTypeChecker
J_PSP_amplitude_matrix = np.block([
    [np.ones((number_excitatory_cells, number_excitatory_cells))*J_PSP_amplitude_excitatory,  # exc --> exc
     np.ones((number_excitatory_cells, number_inhibitory_cells))*-g_exc*J_PSP_amplitude_excitatory],  # inh --> exc
    [np.ones((number_inhibitory_cells, number_excitatory_cells))*J_PSP_amplitude_inhibitory,  # exc --> inh
     np.ones((number_inhibitory_cells, number_inhibitory_cells))*-g_inh*J_PSP_amplitude_inhibitory]  # inh --> inh
])
weighted_connectivity_matrix = connectivity_matrix * J_PSP_amplitude_matrix

# simulation properties
simulation_time = 100  # ms
time_step = 1  # ms

transmission_delay_steps = round(transmission_delay/time_step)

spike_skip_steps = round(refractory_period/time_step)
if spike_skip_steps == 0:
    spike_skip_steps = 1
number_of_time_steps = round(simulation_time/time_step)
time_array = np.arange(0, simulation_time, time_step)
V_t = np.zeros((number_of_cells, number_of_time_steps))  # voltage at each time
V_t[:, 0] = EL  # the voltage for the cells has a potential at t=0 equal to EL
I_t = np.zeros(V_t.shape)  # if Rm is in M ohm, then I_t is in nA. (Current at each time)

start_spike_generation = time.time()
print("Parameter initialisation: "+str(start_spike_generation-start_program)+" s")
external_generated_spike_times = uniform_log_spike_generation(external_freq_excitatory, simulation_time, time_step,
                                                              number_of_time_steps, number_of_cells, C_ext_connections)
end_spike_generation = time.time()
print("External spike generation: "+str(end_spike_generation - start_spike_generation)+" s")

k = 0  # index for the timestep.
spikes = {}  # emtpy dict for the spikes with the key the time step (not time value) and the value the spikes cell(s)
number_of_progression_updates = 10
print_progression_every = round(number_of_time_steps/number_of_progression_updates)
print("Start simulation:")
start_simulation = time.time()
while k < number_of_time_steps-1:
    # cells with cell dynamics which also spiked (therefore have no dynamics next timestep)
    cells_spiked = np.logical_and(np.invert(np.isnan(V_t[:, k])), (V_t[:, k] >= V_th))
    indices_cells_spiked = np.where(cells_spiked)[0]
    if k % print_progression_every == 0:
        progress = k // print_progression_every
        print(str(progress*100/number_of_progression_updates)+"%")

    if indices_cells_spiked.size != 0:
        spikes[k] = indices_cells_spiked

    if (k + spike_skip_steps) < len(V_t[0, :]) - 1:
        V_t[cells_spiked, k:(k + spike_skip_steps)] = np.NaN
        V_t[cells_spiked, (k + spike_skip_steps)] = V_reset

    # determine injected current
    spiked_external_synapses_each_cell = np.count_nonzero(external_generated_spike_times == k, axis=(1, 2))
    I_t[:, k] = 1/Rm*tau_vector*J_PSP_amplitude_excitatory*spiked_external_synapses_each_cell
    spiked_connected_result = np.zeros(weighted_connectivity_matrix.shape)
    if k-transmission_delay_steps in spikes:
        if spikes.get(k-transmission_delay_steps).size != 0:
            spikes_cells_boolean = np.zeros(number_of_cells, dtype=bool)
            spikes_cells_boolean[spikes.get(k-transmission_delay_steps)] = True
            # spikes
            spiked_connected_result = weighted_connectivity_matrix * spikes_cells_boolean
            # ^ gives synaptic weight if both a spike has occurred in the pre-synaptic cell and
            # a connection is present between the pre-synaptic cell and the postsynaptic cell

    # cells with cell dynamics (from this timestep)
    cells_dynamics = np.invert(np.isnan(V_t[:, k]))
    cells_dynamics_matrix = np.logical_and(cells_dynamics[:, None], cells_dynamics[None, :])
    num_cells_dynamics = np.sum(cells_dynamics)
    dynamics_shape = (num_cells_dynamics, num_cells_dynamics)

    # add synaptic currents to injected current to get total current going into the particular cells.
    I_t[cells_dynamics, k] += 1/Rm*tau_vector[cells_dynamics] * \
                              np.sum(spiked_connected_result[cells_dynamics_matrix].reshape(dynamics_shape), axis=1)
    # standard dynamics, when a spike has NOT recently occurred. (Implicit/Backward Euler)
    V_t[cells_dynamics, k + 1] = ((tau_vector[cells_dynamics] * V_t[cells_dynamics, k] +
                                   time_step * EL + Rm * I_t[cells_dynamics, k]) / (tau_vector[cells_dynamics] + time_step))
    # Forward Euler
    # V_t[cells_dynamics, k + 1] = (1 - time_step / tau_vector[cells_dynamics]) * V_t[cells_dynamics, k] + \
    #                              time_step/tau_vector[cells_dynamics] * (Rm * I_t[cells_dynamics, k])
    k += 1

end_simulation = time.time()
print("Simulation: "+str(end_simulation - start_simulation)+" s")
# print(np.nanmean(V_t[0:5, 50:]))
print("End simulation, start plotting")

fig_ext_spikes, ax_ext_spikes = plt.subplots()
ax_ext_spikes.hist(external_generated_spike_times.flatten(), bins=number_of_time_steps + 1)
ax_ext_spikes.set_xlim(0, simulation_time)
ax_ext_spikes.set_xlabel("spike times (ms)")
ax_ext_spikes.set_ylabel("total spike count")

fig_ext_spikes_count, ax_ext_spikes_count = plt.subplots()
spike_count_per_connection_ext = np.count_nonzero(external_generated_spike_times <= number_of_time_steps, axis=2)
# probability_spike_count = spike_count_per_connection_ext/np.sum(spike_count_per_connection_ext)*100
ax_ext_spikes_count.hist(spike_count_per_connection_ext.flatten(), bins=10, density=True)
ax_ext_spikes_count.set_xlabel("spike count per connection")
ax_ext_spikes_count.set_ylabel("probability (%)")
ax_ext_spikes_count.yaxis.set_major_formatter(PercentFormatter(1.0))

fig_ext_spikes_dif, ax_ext_spikes_dif = plt.subplots()
ax_ext_spikes_dif.hist(np.diff(external_generated_spike_times).flatten(), bins=number_of_time_steps + 1, density=True)
ax_ext_spikes_dif.set_xlabel("spike interval (ms)")
ax_ext_spikes_dif.set_ylabel("probability (%)")
ax_ext_spikes_dif.yaxis.set_major_formatter(PercentFormatter(1.0))

number_of_plotted_cells = 5
colors = cm.rainbow(np.linspace(0, 1, number_of_plotted_cells))
line_styles = lines.lineStyles.keys()

fig, ax = plt.subplots()

number_of_scatter_plot_cells = 50
jet_colors = cm.rainbow(np.linspace(0, 1, number_of_scatter_plot_cells))
chosen_cells = rng.choice(number_of_cells, number_of_scatter_plot_cells)
for spike_time, spiked_cells in spikes.items():
    spiked_chosen_cells = np.intersect1d(spiked_cells, chosen_cells)
    if spiked_chosen_cells.size != 0:
        spiked_chosen_cells_indices = np.where(np.in1d(chosen_cells, spiked_chosen_cells))[0]
        ax.scatter(time_array[spike_time]*np.ones(spiked_chosen_cells_indices.shape),
                   spiked_chosen_cells_indices, color=jet_colors[spiked_chosen_cells_indices])
ax.set_yticks([])
ax.set_xlim(0, simulation_time)
ax.set_xlabel('time (ms)')
ax.set_ylabel('cell index')

fig_volt, ax_volt = plt.subplots()
for post_cell_index, post_cell_color in zip(range(number_of_plotted_cells), colors):
    ax_volt.plot(time_array, V_t[post_cell_index, :], color=post_cell_color, label=post_cell_index)
# ax_volt.plot(time_array, V_t[0, :].T, color=colors[0], label=0)
ax_volt.set_ylim(-5, 35)
ax_volt.axhline(V_th, color='r', linestyle=':')
ax_volt.set_xlabel('time (ms)')
ax_volt.set_ylabel('membrane potential (mV)')
ax_volt.legend()

end_plotting = time.time()
print("Plotting: "+str(end_plotting-end_simulation)+" s")
plt.show()
