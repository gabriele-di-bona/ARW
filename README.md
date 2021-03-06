# *Maximal diffusivity of adaptive random walks* (by G. Di Bona, L. Di Gaetano, V. Latora, and F. Coghi)
Implementation of adaptive random walk (ARW), unbiased random walk (URW), and maximum entropy random walk (MERW) on complex networks. In this folder you can find all the code, scripts and notebooks to run the simulations and analysis them as done in our paper.

# Installation
1. Clone this directory to your computer in a new folder.
1. In order to ensure that everything is working as intended, create a dedicated environment using the specified requirements file, using:
    ```conda env create -f ARW.yml```
    *ACHTUNG*: If you want to specify a specific install path rather than the default for your system, just use the -p flag followed by the required path, e.g.:
    ```conda env create -f ARW.yml -p /home/user/anaconda3/envs/ARW```

# How to use
In codes/functions.py there are all the basic functions needed to create or load graphs, and to run simulations.

Currently the only algorithms supported are URW, MERW and ARW. If for example you want to simulate an ARW, you need to do two things:
1. Load a graph, using the function *load_graph* with the parameters you want.
    - ACHTUNG: at the moment, the only graphs supported are Erdos-Renyi and Barabasi-Albert. If the requested graph is not already present in the folder, it is created from scratch, and its largest connected component is dumped and returned.
    - There is also support for an Airport network, i.e., a network between worldwide cities that have a direct flight, but you need to download the files from the owner of the dataset, i.e. from http://seeslab.info/downloads/air-transportation-networks/. Accessing this network from the file *global-net.dat* on 01/01/2022, it is made of 3618 nodes and 14142 links. If you use this network, remember to cite their work (see link)
1. Run all_RW_preprocessing_entropy_theoretic with the parameters you want, specifying ARW as the algorithm.
    - *ACHTUNG*: the function all_RW_preprocessing_entropy_theoretic to run the process uses numba.njit, which speeds up the process. However, if you want to change the code under njit, beware that not all python functions are supported, and you need to be consistent with the types. See https://numba.pydata.org/numba-doc/latest/user/index.html.
    - Notice also that the function is slowed down (a lot) if you do prints under njit. Moreover, if you put a print inside a njit function, the print might appear in output after some time.

## Scripts
There are some scripts you can use to do all the simulations needed to make the plots in the paper. These are the scripts starting with *fig1* or *fig2* inside the folder *codes*. These scripts need the proper arguments. If you have a cluster, you can use the *bash_scripts/* files, in which all parameters and the correct codes are properly set. If you are running the scripts in your local machine, you can use the same bash scripts for a guide to run all the necessary files in the correct way.

If you do not have enough computational resources, you can reduce the number of iterations for each set of parameters and algorithm. Moreover, you can create new scripts that use more than one core at a time, doing more simulations within the same process, by changing appropriately the parameters and, eventually, the code.

All created graphs are stored under *graphs/*, while all simulation results are stored in the proper subdirectories under *data/*, categorized depending on the parameters chosen and

You can also play more with the codes and create new sets of parameters to study new things.

## Analysis
After all needed simulations are done, you can do the analysis and create the plots using *notebooks/plots_analysis.ipynb*. If you have done all simulations correctly, you should be able to run the whole notebook flawlessly, and the figures will be created under *figures/*. 

# Reference
If you use any of the material here provided, you must cite our work using the following bibtex entry:

```
@article{dibona2022maximal,
    title = {Maximal diffusivity of adaptive random walks},
    abstract = {Random walks are the most versatile tool to explore a complex network. These include maximum entropy random walks (MERWs), which are maximally dispersing and therefore play a key role as they optimize information spreading. However, building a MERW comes at the cost of knowing beforehand the global structure of the network to be explored. Here, we propose an adaptive random walk (ARW), which instead maximizes dispersion by updating its transition rule on the local information collected while exploring the network. We show how to derive the ARW via a large-deviation representation of the MERW as a finite-time rare event of an unbiased random walk and study its dynamics on synthetic and real-world networks.},
    author = {Di Bona, Gabriele and Di Gaetano, Leonardo and Latora, Vito and Coghi, Francesco},
    journal = {arXiv preprint arXiv:2202.13923},
    archivePrefix = {arXiv},
    arxivId = {2202.13923},
}
```

# Contact
If you need to contact us for any reason, please do not hesitate to send an email to francesco.coghi@su.se and, especially for code related reasons, to g.dibona@qmul.ac.uk.

Thank you for the interest and for using this repository,

*The authors*
