import json

int_for_random_generator = 42

# network properties
number_of_cells = 12500
ratio_excitatory_cells = 0.8
gamma_ratio_connections = 0.25
epsilon_connection_probability = 0.1  # C_random_excitatory_connections/number_excitatory_cells

# cell properties https://link.springer.com/article/10.1023/A:1008925309027
EL = 0  # mV
V_reset = 10  # mV
Rm = 1  # M ohm (not needed anymore since RI is defined in completeness
tau_E = 20  # ms (time constant RC of excitatory neuron)
tau_I = 20  # ms (time constant RC of inhibitory neuron)
V_th = 20  # mV (threshold voltage (=θ))
refractory_period = 2.0  # ms
transmission_delay = 1.5  # ms

# synaptic properties, and external frequency
J_PSP_amplitude_excitatory = 0.1  # mV (PSP is postsynaptic potential amplitude)
ratio_external_freq_to_threshold_freq = 0.9
g_inh = 4.5  # unitless (so not conductances here) (relative strength of inhibitory connections)

# simulation properties
simulation_time = 600  # ms
time_step = 0.1  # ms

save_voltage_data_every_ms = 1  # ms  (the time between datapoints for the voltage data)
number_of_progression_updates = 10  # the number of updates given (e.g. if it is 10, it will give 0%, 10%, ... , 90%)
number_of_scatter_plot_cells = 50

parameters_dict = {
    'int_for_random_generator': int_for_random_generator,

    'number_of_cells': number_of_cells,
    'ratio_excitatory_cells': ratio_excitatory_cells,
    'gamma_ratio_connections': gamma_ratio_connections,
    'epsilon_connection_probability': epsilon_connection_probability,

    # cell properties https://link.springer.com/article/10.1023/A:1008925309027
    'EL': EL,
    'V_reset': V_reset,
    'Rm': Rm,
    'tau_E': tau_E,
    'tau_I': tau_I,
    'V_th': V_th,
    'refractory_period': refractory_period,
    'transmission_delay': transmission_delay,

    # synaptic properties, and external frequency
    'J_PSP_amplitude_excitatory': J_PSP_amplitude_excitatory,
    'ratio_external_freq_to_threshold_freq': ratio_external_freq_to_threshold_freq,
    'g_inh': g_inh,

    # simulation properties
    'simulation_time': simulation_time,
    'time_step': time_step,

    'save_voltage_data_every_ms': save_voltage_data_every_ms,
    'number_of_progression_updates': number_of_progression_updates,
    'number_of_scatter_plot_cells': number_of_scatter_plot_cells
}

with open("config.json", "w") as configfile:
    json.dump(parameters_dict, configfile)

