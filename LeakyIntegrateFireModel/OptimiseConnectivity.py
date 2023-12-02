import matplotlib.pyplot as plt
from matplotlib import lines
import matplotlib.cm as cm
import numpy as np
from math import ceil
from matplotlib.ticker import PercentFormatter
import time
from FunctionsExpandedBrunelNetwork import *
import scipy.io
from ray import train, tune
from ray.train import RunConfig
from ray.tune.search.optuna import OptunaSearch
from functools import partial
import os
import json


fixed_hyperparameters = {
    "N_total_nodes": 10000,
    "int_random_generator": 42,
    "weight": 0.5,
    "option": "manc",
    # "option": "hemibrain",
}
option = fixed_hyperparameters["option"]

if option == "manc":
    with open('manc_v1.0_indegrees.npy', 'rb') as indegree_file:
        in_degree_elements = np.load(indegree_file)

    with open('manc_v1.0_outdegrees.npy', 'rb') as outdegree_file:
        out_degree_elements = np.load(outdegree_file)
elif option == "hemibrain":
    with open('hemibrain_v1.2.1_indegrees.npy', 'rb') as indegree_file:
        in_degree_elements = np.load(indegree_file)

    with open('hemibrain_v1.2.1_outdegrees.npy', 'rb') as outdegree_file:
        out_degree_elements = np.load(outdegree_file)
else:
    print("The chosen option", option, "is not one of the possible options.")
    exit(1)


m_0_nodes_choice = [50, 100]  # equivalent to "m" in the web application?
rho_probability_choice = [0.1]  # not present in web application? (Noise factor?, Delta?)
l_cardinality_partitions_choice = [10]  # Likely L in the web application
E_k_choice = [-40, -20, 20, 40]  # likely EK in the web application
phi_U_probability_choice = [0.9, 1.0]  # likely phi_U in the web application
phi_D_probability_choice = [0, 0.0005]  # likely phi_D in the web application
delta = [0.1, 0.5, 1.0]
noise_factor = [0.1, 0.5, 1.0]
dimensions = [[1, 0, 1, 0, 1, 0]]
pc = [300]

# web application: https://gtalg.ebrains-italy.eu/connect/

search_space = {
    # "m_0": tune.grid_search(m_0_nodes_choice),
    "m_0": tune.randint(1, 100),
    # "rho": tune.grid_search(rho_probability_choice),
    "rho": tune.choice(rho_probability_choice),
    # "l": tune.grid_search(l_cardinality_partitions_choice),
    "l": tune.choice(l_cardinality_partitions_choice),
    # "E_k": tune.grid_search(E_k_choice),
    "E_k": tune.uniform(-50, 50),
    # "phi_U": tune.grid_search(phi_U_probability_choice),
    "phi_U": tune.uniform(0.8, 1.0),
    # "phi_D": tune.grid_search(phi_D_probability_choice),
    "phi_D": tune.uniform(0, 0.005),
    # "delta": tune.grid_search(delta),
    "delta": tune.uniform(0.0, 5.0),
    # "noise_factor": tune.grid_search(noise_factor),
    "noise_factor": tune.uniform(0.0, 5.0),
    # "dimensions": tune.grid_search(dimensions),
    "dimensions": tune.choice(dimensions),
    # "pc": tune.grid_search(pc)
    "pc": tune.choice(pc)
    # "pc": tune.randint(10, 300)
}
rng = np.random.default_rng(fixed_hyperparameters["int_random_generator"])

start_fitting = time.time()

tuner = tune.Tuner(partial(produce_connectivity_calculate_cross_entropy, fixed_hyperparameters["N_total_nodes"],
                           fixed_hyperparameters["int_random_generator"], fixed_hyperparameters["weight"],
                           in_degree_elements, out_degree_elements), param_space=search_space,
                   tune_config=tune.TuneConfig(
                       num_samples=500,
                       search_alg=OptunaSearch(),
                       metric="score",
                       mode="min",
                   ))

results = tuner.fit()
end_fitting = time.time()
print("Fitting time:", end_fitting-start_fitting)
# best_tunable_hyperparameters = results.get_best_result(metric="score", mode="min").config
# print(type(best_tunable_hyperparameters))
# print(best_tunable_hyperparameters)
# print(results.get_best_result(metric="score", mode="min").metrics["score"])
all_results = results.get_dataframe().sort_values(by=["score"])
top3_results = all_results.iloc[:3, :]
print(top3_results)

folder_name = "saved_data_optimised_connectivity"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

folder_identifier = 0
while os.path.exists(folder_name + "/" + str(folder_identifier)):
    folder_identifier += 1
str_identifier = folder_name + "/" + str(folder_identifier)
os.makedirs(str_identifier)
file_name = str_identifier + "/top3_results.pkl"
top3_results.to_pickle(file_name)

with open(str_identifier + "/fixed_hyperparameters.json", "w") as fixed_hyperparameters_file:
    json.dump(fixed_hyperparameters, fixed_hyperparameters_file)
# in_degree_elements_matlab_sim = out_degree_elements_matlab_sim = [0]
# in_degree_online_probs = out_degree_online_probs = [0]
#
# hist_plot_data_model_degree_distributions(option, in_degree_elements, in_degree_elements_model,
#                                           in_degree_elements_matlab_sim, out_degree_elements, out_degree_elements_model,
#                                           out_degree_elements_matlab_sim)
#
# step_plot_data_model_degree_distributions(option, in_degree, in_degree_gen_model, in_degree_online_probs,
#                                           out_degree, out_degree_gen_model, out_degree_online_probs)
#
# plt.show()