# BrunelNetwork and connectivity
Implementation of Brunel Network [[1]](#1) and an optimisation procedure to generate connectivities using the convolutive model [[2]](#2)[[3]](#3)[[4]](#4). Be aware that the Python scripts assume a working directory one above the [microcircuit_model](microcircuit_model), import statements will need adjustments otherwise. Use conda and the [environment.yml](microcircuit_model/environment.yml) to generate a conda env with the following line: 
```
conda env create -f environment.yml
```
## Simulation
To run the Brunel Network and analyze the results, there are 3 steps:
1. Run [set_parameters_brunel_network.py](microcircuit_model/simulation/set_parameters_brunel_network.py) to create a file (config.json) with the right parameters.
2. Run [brunel_network.py](microcircuit_model/simulation/brunel_network.py) to run the simulation, where it should create a folder: "saved_data_brunel_network" within which there is another folder with an integer as a name (0 if it is run for the first time) containing the saved data.
3. Run [plot_brunel_network.py](microcircuit_model/simulation/plot_brunel_network.py) which needs to be edited such that "str_identifier" at the start of the file refers to the (integer) folder with the saved data created in the previous step.

## Connectivity
To obtain experimental data used in the code:
1. For the option pyc-pyc, download [pyc-pyc subgraph](https://doi.org/10.5281/zenodo.5579388) from [[5]](#5). 
1. Replace "token" in [extract_data.py](microcircuit_model/connectivity/data_preparation/extract_data.py) with your token. You can obtain your authorization token by following the steps in the quickstart guide at (https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html). Next, run [extract_data.py](microcircuit_model/connectivity/data_preparation/extract_data.py) in [data_preparation](microcircuit_model/connectivity/data_preparation) to acquire the degree elements used for optimisation. (the two options `"manc"` and `"hemibrain"` should work directly).

After obtaining experimental data optimised connectivity can be created:
1. Run [optimise_connectivity.py](microcircuit_model/connectivity/optimise_connectivity.py) where the `option` in `fixed_hyperparameters` should be the same one used when you generated the raw data. This creates a folder "saved_data_optimised_connectivity" within which there is a folder with an integer as a name, where it saves the data to recreate the top 3 results.
2. Run [analyze_optimised_connectivity.py](microcircuit_model/connectivity/analyze_optimised_connectivity.py) with specific functions uncommented to analyse the optimised connectivities differently. Notably, `str_identifier` should refer to the (integer) folder(s) created when you ran [optimise_connectivity.py](microcircuit_model/connectivity/optimise_connectivity.py).
3. (In case the function save_parameters_top_result was used in [analyze_optimised_connectivity.py](microcircuit_model/connectivity/analyze_optimised_connectivity.py) the parameters of the top result can be used more easily to obtain generated connectivities using [get_optimised_connectivity.py](microcircuit_model/connectivity/get_optimised_connectivity.py))

## References
<a id="1">[1]</a> 
Brunel, N. Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons. *Journal of Computational Neuroscience* **8**, 183–208. ISSN: 1573-6873. (https://doi.org/10.1023/A:1008925309027) (2023) (May 1, 2000)

<a id="2">[2]</a> 
Giacopelli, G., Tegolo, D., Spera, E. & Migliore, M. On the structural connectivity of large-scale models of brain networks at cellular level. _Scientific Reports_ **11**. Number: 1 Publisher: Nature Publishing Group, 4345. ISSN: 2045-2322. (https://www.nature.com/articles/s41598-021-83759-z) (2023) (Feb. 23, 2021).

<a id="3">[3]</a> 
Giacopelli, G., Migliore, M. & Tegolo, D. Graph-theoretical derivation of brain structural connectivity. _Applied Mathematics and Computation_ **377**, 125150. ISSN: 0096-3003. (https://www.sciencedirect.com/science/article/pii/S0096300320301193) (2023) (July 15, 2020).

<a id="4">[4]</a> 
Giacopelli, G., Migliore, M. & Tegolo, D. Spatial graphs and Convolutive Models.

<a id="5">[5]</a>
Turner, N. L. et al. Reconstruction of neocortex: Organelles, compartments, cells, circuits, and activity. _Cell_ **185**, 1082–1100.e24. ISSN: 0092-8674. (https://www.sciencedirect.com/science/article/pii/S0092867422001349) (2024) (Mar. 17, 2022).
