import numpy as np
import networkx as nx
import joblib
import os
import datetime
import multiprocessing as mp
import argparse

from functions import *
set_params_figures()


start_time = datetime.datetime.now()
print(f"Started at {start_time}", flush = True)
# Change directory to the root of the folder (this script was launched from the subfolder codes)
os.chdir("../")

# Saving in a particular folder
output_dir = "./data/different_N_no_renormalization/" 
os.makedirs(output_dir,exist_ok=True)

# Use arguments to give from shell
parser = argparse.ArgumentParser(description='Run simulations of URW, MERW and ARW for Erdos_Renyi, Barabasi_Albert, and Airports networks.')

parser.add_argument("-ID", "--ID", type=int,
    help="ID to get the correct the simulation parameters. \
          If more cores are available, it does the simulations in the range(ID*cores, (ID+1)*cores), \
          so always one for each core [default: 1]",
    default=1)
parser.add_argument("-c", "--cores", type=int,
    help="Number of cores available for the job [default: 1]",
    default=1)

arguments = parser.parse_args()
ID = arguments.ID
cores = arguments.cores


# Save info for retrieving the graph
graphs = {}
graph_dir = "./graphs/"
N_requested = 1000
# I can add a number to the keyword Erdos_Renyi, because it gets the correct name by selecting only letters and giving back Erdos_Renyi
for N_requested in sorted(set(list(np.geomspace(1, 10000, 1000, dtype=np.int64)))): # this has len 599
    graphs["Erdos_Renyi_%d"%N_requested] = {"N_requested":N_requested, "avg_k_requested":3., "graph_dir":graph_dir, "max_tries":100}

# PARAMETERS
algorithms = ["ARW"]
alpha = 0.1
chosen_dts = [int(1e13)]
num_to_save = 2
types_entropy = [2,3]
stop_at_covering_links = True
renormalize_r = False
seed = np.random.randint(2**32-1)
max_M = 0

print(f"For each repetition, %d simulations are to be done"%(len(graphs) * len(algorithms) * len(chosen_dts)), flush=True)


print("ID in range(%d, %d)"%(ID*cores, (ID+1)*cores))

if cores > 1:
    pool = mp.Pool(cores)
    for i in range(ID*cores, (ID+1)*cores):
        pool.apply_async(do_multiple_parameters_simulation, args=(ID,graphs, algorithms, alpha, chosen_dts, num_to_save, types_entropy, stop_at_covering_links, output_dir, renormalize_r, seed, max_M))
    pool.close()
    pool.join()
else:
    for i in range(ID*cores, (ID+1)*cores):
        do_multiple_parameters_simulation(ID,graphs, algorithms, alpha, chosen_dts, num_to_save, types_entropy, stop_at_covering_links, output_dir, renormalize_r, seed, max_M)
