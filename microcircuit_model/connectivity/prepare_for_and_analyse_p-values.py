# import rpy2
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr, data
import numpy as np

from data_preparation.data_prep_utils import *
from connectivity_utils import *
import os.path
import pandas as pd

# neuprint documentation: https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html
# option = "manc"  # manc --> "https://www.janelia.org/project-team/flyem/manc-connectome"
# option = "hemibrain"  # hemibrain --> "https://www.janelia.org/project-team/flyem/hemibrain"
option = "pyc-pyc"
# from MICrONS Explorer https://www.microns-explorer.org/phase1, raw data at: https://zenodo.org/records/5579388
# accompanying paper: Turner, N. L., Macrina, T., Bae, J. A., Yang, R., Wilson, A. M., Schneider-Mizell, C., Lee, K.,
# Lu, R., Wu, J., Bodor, A. L., Bleckert, A. A., Brittain, D., Froudarakis, E., Dorkenwald, S., Collman, F., Kemnitz,
# N., Ih, D., Silversmith, W. M., Zung, J., … Seung, H. S. (2022). Reconstruction of neocortex: Organelles,
# compartments, cells, circuits, and activity. Cell, 185(6), 1082-1100.e24. https://doi.org/10.1016/j.cell.2022.01.023

# network_sizes = [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12500, 15000, 17500, 20000]
network_sizes = [6, 10, 20, 30, 40, 50, 82, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

# if "create_csv_degree_elements_not_analyse_p_values" is True: create the .csv files with degree elements.
# If False: analyse the p-values (cannot do both at the same time,
# since the R script "calculate_ks_stats.R" must be run to create the files with p-values.)
create_csv_degree_elements_not_analyse_p_values = True

# prepare the data by creating connectivity's and saving the degree elements into .csv files.
if create_csv_degree_elements_not_analyse_p_values:
    in_degree_elements, out_degree_elements = get_degree_data(option)
    to_csv_degree_elements(in_degree_elements, out_degree_elements, option)
    for network_size_creation in network_sizes:
        create_and_save_model_csv_degree_elements(option, 10, network_size_creation)

# analyse the p-value generated from the R script "calculate_ks_stats.R"
if not create_csv_degree_elements_not_analyse_p_values:

    for total_nodes in network_sizes:
        # analyse p-values
        # total_nodes = 1000
        base_folder = "ks_stats/" + option + "/"
        complete_filename = base_folder + "ks_stats_" + str(total_nodes) + ".csv"
        ks_stats_dataframe = pd.read_csv(complete_filename)
        p_values_indegree = np.array(ks_stats_dataframe["p_value_indegree"])
        adjusted_p_values_indegree = 2 * np.minimum(p_values_indegree, 1 - p_values_indegree)
        p_values_outdegree = np.array(ks_stats_dataframe["p_value_outdegree"])
        adjusted_p_values_outdegree = 2 * np.minimum(p_values_outdegree, 1 - p_values_outdegree)

        # calculate mean and standard deviation of p-values
        mean_p_value_indegree = np.average(adjusted_p_values_indegree)
        std_p_value_indegree = np.std(adjusted_p_values_indegree, ddof=1)
        mean_p_value_outdegree = np.average(adjusted_p_values_outdegree)
        std_p_value_outdegree = np.std(adjusted_p_values_outdegree, ddof=1)

        # get bins of size 0.05
        bin_size = 0.05
        num_bins = round(1/bin_size)
        num_bin_edges = 1 + num_bins
        bin_edges = np.linspace(0, 1, num_bin_edges)
        print(bin_edges)
        p_value_indegree_counts, indegree_bin_edges = np.histogram(adjusted_p_values_indegree, bins=bin_edges)
        p_value_outdegree_counts, outdegree_bin_edges = np.histogram(adjusted_p_values_outdegree, bins=bin_edges)
        print(p_value_indegree_counts, p_value_outdegree_counts)
        normalized_p_value_indegree_counts = p_value_indegree_counts/np.sum(p_value_indegree_counts)
        normalized_p_value_outdegree_counts = p_value_outdegree_counts/np.sum(p_value_outdegree_counts)
        print(normalized_p_value_indegree_counts, normalized_p_value_outdegree_counts)
        if normalized_p_value_indegree_counts[0] > bin_size:
            print("It is unlikely that the model indegree distributions are samples from the experimental distribution")
            same_indegree_distribution = False
        else:
            same_indegree_distribution = True

        if normalized_p_value_outdegree_counts[0] > bin_size:
            print("It is unlikely that the model outdegree distributions are samples from the experimental distribution")
            same_outdegree_distribution = False
        else:
            same_outdegree_distribution = True
        statistics_dict = {
            "mean_p_value_indegree": mean_p_value_indegree,
            "std_p_value_indegree": std_p_value_indegree,
            "mean_p_value_outdegree": mean_p_value_outdegree,
            "std_p_value_outdegree": std_p_value_outdegree,
            "likely same indegree distribution": same_indegree_distribution,
            "likely same outdegree distribution": same_outdegree_distribution,
            "indegree": list(normalized_p_value_indegree_counts),
            "outdegree": list(normalized_p_value_outdegree_counts),
            "bin_edges": list(bin_edges),
        }
        with open(base_folder + "p_values_histogram_and_conclusion_" + str(total_nodes) + ".json", "w") as stat_file:
            json.dump(statistics_dict, stat_file)

