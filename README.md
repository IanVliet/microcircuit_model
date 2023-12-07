# BrunelNetwork and connectivity
Manual implementation of Brunel Network and a convolutional model for network connectivity
## Brunel Network
To run the Brunel Network and observe the results, there are 3 steps:
1. Run set_parameters_BrunelNetwork.py to create a file (config.json) with the right parameters.
2. Run BrunelNetwork.py to run the simulation, where it should create a folder: "saved_data_brunel_network" within which there is another folder with an integer as name (0 if it is run for the first time) containing the saved data.
3. Run plotBrunelNetwork.py which needs to be edited such that "str_identifier" at the start of the file refers to the correct folder created in the previous step.

## Optimise Connectivity
To create optimised connectivity there are three steps:
1. Run extractDataNeuPrint.py to acquire raw date (for now there are two options "manc" and "hemibrain").
2. Run OptimiseConnectivity.py where the "option" in "fixed_hyperparameters" should be the same one used when you generated the raw data.
3. Run AnalyzeOptimisedConnectivity.py where "str_identifier" should refer to the specific folder created when you ran OptimiseConnectivity.py.
