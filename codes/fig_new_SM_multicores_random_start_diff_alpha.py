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

parser.add_argument("-a", "--alpha", type=float,
    help="Annealing schedule parameter alpha, choose a float between 0.01 and 1 [default: 0.1]",
    default=0.1)

arguments = parser.parse_args()
ID = arguments.ID
cores = arguments.cores
alpha = arguments.alpha

# Saving in a particular folder
output_dir = "./data/diff_new_alpha/"#"%.4f/"%(alpha) 
os.makedirs(output_dir,exist_ok=True)

# Save info for retrieving the graph
graphs = {}
graph_dir = "./graphs/"
N_requested = 1000
for i,avg_k_requested in enumerate(set([4,5,7]).union(set(np.linspace(9, 30, 8)))):
    graphs[f"Erdos_Renyi_{i}"] = {"N_requested":N_requested, \
                             "avg_k_requested":avg_k_requested, \
                             "graph_dir":graph_dir, \
                             "max_tries":100}
for i,avg_k_requested in enumerate(set(np.linspace(4, 30, 14))):
    graphs[f"Barabasi_Albert_{i}"] = {"N_requested":N_requested, \
                                 "avg_k_requested":avg_k_requested, \
                                 "graph_dir":graph_dir, \
                                 "max_tries":100}
print(len(graphs))

# PARAMETERS
# do URW and MERW only for alpha=0.1
if alpha == 0.1:
#     algorithms = ["URW", "MERW", "ARW"]
    algorithms = ["ARW"]
else:
    algorithms = ["ARW"]
# alpha = 0.1 # given by command line while submitting the related script
chosen_dts = [int(1e10)]
num_to_save = 100
types_entropy = [3]
stop_at_covering_links = True
renormalize_r = True
seed = np.random.randint(2**32-1)
max_M = 0

print(f"For each repetition, %d simulations are to be done"%(len(graphs) * len(algorithms) * len(chosen_dts)), flush=True)


print("ID in range(%d, %d)"%(ID*cores, (ID+1)*cores))

if cores > 1:
    pool = mp.Pool(cores)
    for i in range(ID*cores, (ID+1)*cores):
        seed = np.random.randint(2**32-1)
        pool.apply_async(do_multiple_parameters_simulation, args=(i,graphs, algorithms, alpha, chosen_dts, num_to_save, types_entropy, stop_at_covering_links, output_dir, renormalize_r, seed, max_M))
    pool.close()
    pool.join()
else:
    for i in range(ID*cores, (ID+1)*cores):
        do_multiple_parameters_simulation(ID,graphs, algorithms, alpha, chosen_dts, num_to_save, types_entropy, stop_at_covering_links, output_dir, renormalize_r, seed, max_M)
