# BrunelNetwork and connectivity
Manual implementation of Brunel Network and a convolutional model for network connectivity. Use conda and the [environment.yml](leaky_integrate_fire_model/environment.yml) to generate a conda env with the following line: 
```
conda env create -f environment.yml
```
## Simulation
To run the Brunel Network and analyze the results, there are 3 steps:
1. Run [set_parameters_brunel_network.py](leaky_integrate_fire_model/simulation/set_parameters_brunel_network.py) to create a file (config.json) with the right parameters.
2. Run [brunel_network.py](leaky_integrate_fire_model/simulation/brunel_network.py) to run the simulation, where it should create a folder: "saved_data_brunel_network" within which there is another folder with an integer as a name (0 if it is run for the first time) containing the saved data.
3. Run [plot_brunel_network.py](leaky_integrate_fire_model/simulation/plot_brunel_network.py) which needs to be edited such that "str_identifier" at the start of the file refers to the (integer) folder with the saved data created in the previous step.

## Connectivity
To create optimised connectivity there are three steps:
1. Run [extract_data.py](leaky_integrate_fire_model/connectivity/data_preparation/extract_data.py) in [data_preparation](leaky_integrate_fire_model/connectivity/data_preparation) to acquire the raw data (for now there are two options `"manc"` and `"hemibrain"`).
2. Run [optimise_connectivity.py](leaky_integrate_fire_model/connectivity/optimise_connectivity.py) where the `option` in `fixed_hyperparameters` should be the same one used when you generated the raw data. This creates a folder "saved_data_optimised_connectivity" within which there is a folder with an integer as a name, where it saves the data to recreate the top 3 results.
3. Run [analyze_optimised_connectivity.py](leaky_integrate_fire_model/connectivity/analyze_optimised_connectivity.py) to recreate the top 3 results, where `str_identifier` should refer to the (integer) folder created when you ran [optimise_connectivity.py](leaky_integrate_fire_model/connectivity/optimise_connectivity.py).
