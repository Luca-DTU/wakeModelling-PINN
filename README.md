Repository for my own work on the special course on Physics Informed Neural Networks for
Wind turbine Wake modelling:
There are three branches in the repository, the interesting ones are:
 - main: contains the fully developed single flow case.
 - big_data : contains the code for the multiflow cases.

In both cases the code runs by executing main.py with the configuration file indicated in the hydra decorator of the my_app function within the script
The configurations are found in the conf folder which allows for both single runs and multiple runs (i.e. sweeps).
The sweeps can be analysed with the script analyse_sweep.py 
hpc submission follow the bash script jobscript.sh
