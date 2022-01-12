#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
import datetime
import joblib
import networkx as nx
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import normalize
from numba import njit
import numba

#%%
def set_params_figures(figsize=(8,5), labelsize=22, titlesize=25, ticks_labelsize=18, legend_fontsize=18):
    '''
        Default parameters in the figures
        - figsize = (8,5)
        - labelsize = 22
        - titlesize = 25
        - ticks_labelsize = 18
        - legend_fontsize = 18
    '''
    plt.style.use("default")

    params_default = {
        # no upper and right axes
        'axes.spines.right' : False,
        'axes.spines.top' : False,
        # no frame around the legend
        "legend.frameon" : False,

        # dimensions of figures and labels
        # we will play with these once we see how they are rendered in the latex
        'figure.figsize' : figsize,

        'axes.labelsize' : labelsize,
        'axes.titlesize' : titlesize,
        'xtick.labelsize' : ticks_labelsize,
        'ytick.labelsize' : ticks_labelsize,
        'legend.fontsize' : legend_fontsize,
        # no grids
        'axes.grid' : False,

        # the default color(s) for lines in the plots: in order if multiple lines. We can change them or add colors if needed
    #     'axes.prop_cycle' : mpl.cycler(color=["#00008B", "#BF0000", "#006400"]), 

        # default quality of the plot. Not too high but neither too low
        "savefig.dpi" : 300,
        "savefig.bbox" : 'tight', 

    }

    plt.rcParams.update(params_default)
    plt.rcParams['image.cmap'] = 'gray'


def find_pattern(pattern, path):
    '''
        Finds all the paths inside path that are of the given pattern.
        
        E.g.: find_pattern("*.pkl", "./") to find all the files terminating with .pkl inside the current directory
        
        Returns the list of all such paths (empty is none is found).
    '''
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

#%%

def plot_graph(G, graph_dir = os.path.join(os.getcwd(),'Graphs'), 
    use_spring_layout=True, use_circular_layout=False, save_fig=True, fig_name="graph", save_as_pdf=True):
    '''
        Plots a graph G. 

        By default G is plotted with spring layout. Optional layout: circular.
        Also save the figure in graph_dir, named fig_name as a pdf file (save_as_pdf=True) or png (save_as_pdf=False)

        Input
            - G: networkx graph
            - graph_dir = directory where graphs and their plots are stored (default: os.path.join(os.getcwd(),'Graphs') ) 
            - use_spring_layout: make a plot using spring layout of the graph (default: True)
            - use_circular_layout: make a plot using circular layout of the graph (default: False)
            - save_fig: save the plot with use_spring_layout if True, otherwise with use_circular_layout if True (default: True). 
            - fig_name: name of the graph plot to be saved (default: "graph")
            - save_as_pdf: saves the plot as a pdf if True, as png if False (default: True)

        Output
            - None
    '''    
    if use_circular_layout == True:
        plt.figure()
        pos = nx.circular_layout(G)
        nx.draw(G, pos = pos, with_labels=True, font_size=14, node_size=300)
        plt.show()
    if use_spring_layout == True:
        plt.figure()
        pos = nx.spring_layout(G)
        nx.draw(G, with_labels=True, pos=pos, font_size=14, node_size=300)
        plt.show()
    
    if save_fig == True:
        if save_as_pdf == True:
            plt.savefig(os.path.join(graph_dir,fig_name+".pdf"))
        else:
            plt.savefig(os.path.join(graph_dir,fig_name+".png"))
    
    return None


def get_name_graph(name_graph):
    '''
        Takes the string name_graph, tries to understand if it is one of the supported ones and returns it, 
        otherwise returns an empty string.
        
        Supported graphs (lowercase or uppercase is the same):
            - Erdos_Renyi (or Erdos-Renyi or Erdos Renyi)
            - Barabasi_Albert (or Barabasi-Albert or Barabasi Albert)
            - Airports (or airport or global or global-net or global_net)

    '''
    # put in lowercase
    tmp_string = name_graph.lower()
    # select only letters
    only_letters_string = ""
    for l in tmp_string:
        if l in 'qwertyuiopasdfghjklzxcvbnm':
            only_letters_string += l
    # check if it is one of the supported cases
    if only_letters_string == "erdosrenyi" or only_letters_string == "erdosrenji":
        correct_name_graph = "Erdos_Renyi"
    elif only_letters_string == "barabasialbert":
        correct_name_graph = "Barabasi_Albert"
    elif only_letters_string == "airports" or only_letters_string == "airport" or only_letters_string == "global" or only_letters_string == "globalnet":
        correct_name_graph = "Airports"
    else:
        correct_name_graph = ""
    # returns the correct name
    return correct_name_graph


def load_graph(name_graph = "Erdos_Renyi", N_requested = 50, avg_k_requested = 3.00, 
    graph_dir = os.path.join(os.getcwd(),'graphs'), 
    max_tries = 1,
    do_prints = True,
    create_new = True,
    show_graph = False, use_spring_layout=True, use_circular_layout=False, save_fig=False, save_as_pdf=True):
    '''
        Load the requested graph (saved as adj_matrix). If not stored in graph_dir, try to make it and store it.
        In the latter case, it will create it with the name of N and avg_k requested, not the created one.
        Returns the networkx graph and the (dense) adjacency matrix.

        Supported graphs (lowercase or uppercase is the same):
            - Erdos_Renyi (or Erdos-Renyi or Erdos Renyi)
            - Barabasi_Albert (or Barabasi-Albert or Barabasi Albert)
            - Airports (or global or global-net or global_net)

        Input 
            - name_graph = type of graph (default "Erdos_Renyi"). If graph not present, 
            will try to create one if it's "Erdos_Renyi" or "Barabasi_Albert", 
            connected, with that number of nodes and that average degree.
                - In case of Barabasi_Albert graph, 1 <= k <= N/2. 
                In this case, m_0 = m vertices are initialized (no edges) and then
                other N-m vertices are sequentially created with m edges to existing nodes.
                m is calculated from N and k and it will be integer, so the average degree could be different.
            - N = number of nodes in the graph (default: 50). Useful only for Erdos_Renyi and Barabasi_Albert
            - avg_k = average degree in the graph with 2 decimal precision (use %.2f) (default: 3.09). Useful only for Erdos_Renyi and Barabasi_Albert
            - graph_dir = directory where graphs and their plots are stored (default: os.path.join(os.getcwd(),'Graphs') ) 
            - max_tries = number of tries to create the best graph requested, useful only for Erdos_Renyi (default: 1)
            - show_graph = False (if True, plots the graph)
            # Next parameters are useful only if show_graph = True
            - use_spring_layout: make a plot using spring layout of the graph (default: True)
            - use_circular_layout: make a plot using circular layout of the graph (default: False)
            - save_fig: save the plot with use_spring_layout if True, otherwise with use_circular_layout if True (default: True). 
            - save_as_pdf: saves the plot as a pdf if True, as png if False (default: True)

        Output 
            - G: networkx graph representation of requested graph
            - A: dense adjacency matrix of requested graph
    '''
    os.makedirs(graph_dir,exist_ok=True)
    avg_k_requested = float(avg_k_requested)
    # get the correct name of the graph
    name_graph = get_name_graph(name_graph)
    name_file = name_graph
    if name_graph in ["Erdos_Renyi", "Barabasi_Albert"]:
        name_file += "_N_%d_k_%.2f.pkl"%(N_requested,avg_k_requested)
    try:
        dump_graph = False # Become True if this is not found and a new graph is created
        with open(os.path.join(graph_dir, name_file + ".pkl"), 'rb') as fp:
            A = joblib.load(fp)
        G = nx.convert_matrix.from_numpy_matrix(A)
        if do_prints:
            if name_graph in ["Erdos_Renyi", "Barabasi_Albert"]:
                print("Loaded %s with requested N=%d and average degree avg_k=%.02f, actual is N=%d and avg_k=%.02f."%(name_graph,N_requested,avg_k_requested, len(G), np.round(np.mean(list(dict(G.degree()).values())), 2)), flush=True)
            else:
                print("Loaded %s with actual N=%d and avg_k=%.02f."%(name_graph, len(G), np.round(np.mean(list(dict(G.degree()).values())), 2)), flush=True)
    except FileNotFoundError as e:
        if do_prints:
            print("Couldn't load the graph.\n", e, flush=True)
        if create_new == False:
            return None,None
        else:
            dump_graph = True
            if name_graph == "Erdos_Renyi":
                if do_prints:
                    print("Creating Erdos-Renyi graph with N=%d and average degree avg_k=%.02f."%(N_requested,avg_k_requested), flush=True)
                # Creating an Erdos-Renyi graph optimizing the the main connected component maximum
                # and the average degree similar to the provided k
                p = avg_k_requested/(N_requested-1)
                max_tries = max_tries
                largest_cc = [] 
                avg_k = -1
                for it in range(max_tries):
                    G2 = nx.erdos_renyi_graph(N_requested, p)
                    largest_cc2 = max(nx.connected_components(G2), key=len)
                    Gcc = sorted(nx.connected_components(G2), key=len, reverse=True)
                    G2 = G2.subgraph(Gcc[0])
                    avg_k2 = np.round(np.mean(list(dict(G2.degree()).values())), 2)
                    if abs(N_requested-len(largest_cc2)) < abs(N_requested-len(largest_cc)):
                        G = G2.copy()
                        avg_k = avg_k2
                        largest_cc = largest_cc2.copy()
                    elif len(largest_cc2) == len(largest_cc) and abs(avg_k2 - avg_k_requested) < abs(avg_k - avg_k_requested):
                        G = G2.copy()
                        avg_k = avg_k2
                        largest_cc = largest_cc2.copy()
                    elif abs(avg_k2 - avg_k_requested) < 0.01 and len(largest_cc2) == N_requested:
                        G = G2.copy()
                        avg_k = avg_k2
                        largest_cc = largest_cc2.copy()
                        if do_prints:
                            print("Graph found!", flush=True)
                        break

            elif name_graph == "Barabasi_Albert":
                # m must be integer! ==> k must be multiple of 2
                if N_requested**2 - 2*avg_k_requested*N_requested < 0:
                    if do_prints:
                        print("You requested a Barabasi_Albert graph, so 1 <= k <= N/2!!")
                    return None,None
                if do_prints:
                    print("m should be",(N_requested - np.sqrt(N_requested**2 - 2*avg_k_requested*N_requested)) / 2)
                m = int(np.round((N_requested - np.sqrt(N_requested**2 - 2*avg_k_requested*N_requested)) / 2))
                if do_prints:
                    print("Creating Barabasi-Albert graph with N=%d and average degree k=%.02f ==> m = %d."%(N_requested,avg_k_requested,m), flush=True)
                G = nx.barabasi_albert_graph(N_requested, m)
                Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(Gcc[0])
                avg_k = np.round(np.mean(list(dict(G.degree()).values())),2)
                if do_prints:
                    print("Graph found!", flush=True)

            elif name_graph == "Airports":
                # ACHTUNG: global-net.dat has to be in graph directory graph_dir!
                net_df_path = os.path.join(graph_dir, "global-net.dat")
                net_df = pd.read_csv(net_df_path, header=None, sep = ' ')
                net_df.columns = ['City 1','City 2']
                net_df.head(5)

                G = nx.from_pandas_edgelist(net_df, source = 'City 1', target = 'City 2',
                                        create_using = nx.Graph())
                Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(Gcc[0])
                if do_prints:
                    print("Graph found!", flush=True)

            else:
                if do_prints:
                    print("name_graph=%s not recognized, exiting."%graph_dir, flush=True)
                return None,None
    
    A = nx.adjacency_matrix(G).todense() # adjacency matrix
    G = nx.from_numpy_matrix(A) # so that it is surely relabeled from 0 to N-1
    
    if dump_graph == True:
        # dumping the graph adjacency matrix
        avg_k = np.mean(list(dict(G.degree()).values()))
        if do_prints:
            print("Requested N=%d, k=%.2f, created with N=%d, k=%.2f. Namefile dumped with original request."
                %(N_requested,avg_k_requested, len(A), avg_k), flush=True)
        with open(os.path.join(graph_dir, name_file + ".pkl"), 'wb') as fp:
            joblib.dump(A, fp)
    if show_graph == True:
        plot_graph(G, use_spring_layout=use_spring_layout, use_circular_layout=use_circular_layout, 
                    save_fig=save_fig, fig_name=name_graph+" N=%d k=%.02f"%(N_requested,avg_k_requested), save_as_pdf=save_as_pdf)
    return G, A

#%%

@njit
def get_max_eigenvector(matrix, it, left=False, normalized_l1=True, normalized_l2=False):
    '''
        This function returns the maximum eigenvalue and its corresponding right (or left) eigenvector.
        if left == True, then it returns the left eigenvector of the matrix corresponding to the max eigenvector.
        
        ACHTUNG: This supposes that the maximum eigenvalue is unique and simple.
        
        ACHTUNG: choose only one normalization between l1 and l2. If both are positive, l1 is preferred
        
        ACHTUNG: the variable "it" is there only to show the iteration number in which we got the error
    '''
    small_number = 1e-4
    matrix_tmp = matrix.copy()
    if left == True:
        # Let's calculate left eigenvalues/eigenvector as right eigenvectors of the matrix transposed
        matrix_tmp = np.transpose(matrix_tmp)
    matrix_tmp = matrix_tmp.astype(numba.types.complex64)
    eigenvalues,eigenvectors = np.linalg.eig(matrix_tmp)
    # Let's get the max eigenvalue and its corresponding eigenvector
    tmp_max_eig = eigenvalues.max()
    real_eigenvalues = np.real(eigenvalues)
    max_eig = real_eigenvalues.max()
    if np.imag(tmp_max_eig) > small_number:
        print(it,"There is something wrong, max eigenvalue is imaginary:",tmp_max_eig, "changed to",max_eig)
    max_eig_position = np.argmax(real_eigenvalues)
    tmp_max_eigenvector = eigenvectors[:,max_eig_position] # np.array(np.transpose(np.real(eigenvectors)[:,max_eig_position]))#.flatten()
    max_eigenvector = np.real(eigenvectors)[:,max_eig_position] 
    flag_imaginary = False
    for val in tmp_max_eigenvector:
        if flag_imaginary == False and np.imag(val) > small_number:
            print(it,"There is something wrong, max eigenvector has imaginary values, e.g.:",val)
            flag_imaginary = True
    if normalized_l1 == True:
        max_eigenvector /= max_eigenvector.sum()
    elif normalized_l2 == True:
        # By default max_eigenvector is normalized in l2, this is just to ensure it is correct
        max_eigenvector /= np.sqrt(max_eigenvector.dot(max_eigenvector))
    return max_eig, max_eigenvector

@njit
def get_entropy(pi,it):
    '''
        Returns entropy production rate of pi and the maximum one.
        
        ACHTUNG: "pi" needs to be a transition matrix, so l1-normalized by rows
        
        "it" is the number of iteration to show if there is any mistake in the calculation of the maximum eigenvalue, which should be real.
    '''
    max_eig_pi, max_eigenvector_pi = get_max_eigenvector(pi,it,left=True, normalized_l1=True, normalized_l2=False)
    entropy = 0.
    for i in range(len(pi)):
        for j in range(len(pi)):
            if pi[i][j] > 0:
                entropy -= max_eigenvector_pi[i] * pi[i][j] * np.log(pi[i][j])
    # now let's repeat on the adjacency matrix to get the max entropy related to this transition matrix
    adjacency_pi = pi.copy()
    for i in range(len(pi)):
        for j, pi_ij in enumerate(adjacency_pi[i,:]):
            if pi_ij > 0:
                adjacency_pi[i,j] = 1
    max_eig, max_eigenvector = get_max_eigenvector(adjacency_pi,it,left=True, normalized_l1=True, normalized_l2=False)
    if max_eig == 0:
        max_entropy = 0.
    else:
        max_entropy = np.log(max_eig)
    return entropy,max_entropy


@njit
def get_explored_transition_matrix(pi,explored_nodes_list,explored_links_matrix,type_entropy):
    '''
        - type_entropy = 0 ---> entropy calculated on graph of explored nodes and all their links, 
            included first neighbors and their links to the explored nodes
        - type_entropy = 1 ---> entropy calculated on graph of explored nodes and all their links, 
            included first neighbors but considered different (unknown) for each link 
            (multiple instances of the same neighbor for each of the links to the explored nodes)
        - type_entropy = 2 ---> entropy calculated on subgraph of explored nodes, with links only between themselves
        - type_entropy = 3 ---> entropy calculated on the graph of only explored links
    '''
    if type_entropy not in {0,1,2,3}:
        # given type_entropy is not supported, overiding it to 0
        type_entropy = 0
    
    
    if type_entropy == 0:
        explored_and_neighbors_list = list(explored_nodes_list)
        transition_matrix = pi.copy()
        for i in range(len(transition_matrix)):
            if i not in explored_nodes_list:
                for j,pi_ij in enumerate(transition_matrix[i,:]):
                    if pi_ij > 0 and j not in explored_nodes_list:
                        transition_matrix[i,j] = 0
            tmp_sum = transition_matrix[i,:].sum()
            if tmp_sum > 0:
                transition_matrix[i,:] /= tmp_sum
        
    elif type_entropy == 1:
        num_nodes = len(explored_nodes_list)
        neighbor_links = [(0,0,0,0) for i in range(0)] # (new_source,old_source, new_target,old_target)
        for i,node_i in enumerate(explored_nodes_list):
            for node_j,pi_ij in enumerate(pi[node_i,:]):
                if pi_ij > 0 and node_j not in explored_nodes_list:
                    # then node_j is oneof the first neighbors, to be considerate separate
                    neighbor_links.append((i,node_i, num_nodes,node_j))
                    num_nodes += 1
                          
        transition_matrix = np.zeros((num_nodes,num_nodes),dtype=numba.types.float64)
        for i,node_i,j,node_j in neighbor_links:
            transition_matrix[i,j] = pi[node_i,node_j]
            transition_matrix[j,i] = 1 # these rows are already normalized
        for i,node_i in enumerate(explored_nodes_list):
            for j,node_j in enumerate(explored_nodes_list):
                transition_matrix[i,j] = pi[node_i,node_j]
            tmp_sum = transition_matrix[i,:].sum()
            if tmp_sum > 0:
                transition_matrix[i,:] /= tmp_sum
     
    
    elif type_entropy == 2:
        transition_matrix = np.zeros((len(explored_nodes_list),len(explored_nodes_list)),dtype=numba.types.float64)
        for i,node_i in enumerate(explored_nodes_list):
            for j,node_j in enumerate(explored_nodes_list):
                transition_matrix[i,j] = pi[node_i,node_j]
            tmp_sum = transition_matrix[i,:].sum()
            if tmp_sum > 0:
                transition_matrix[i,:] /= tmp_sum
                
    elif type_entropy == 3:
        # explored_links_matrix is useful only here, and must be of the same shape of pi
        if len(explored_links_matrix) != len(pi):
            print("explored_links_matrix is not of the same shape as pi...")
        transition_matrix = np.zeros((len(explored_nodes_list),len(explored_nodes_list)),dtype=numba.types.float64)
        for i,node_i in enumerate(explored_nodes_list):
            for j,node_j in enumerate(explored_nodes_list):
                if explored_links_matrix[node_i,node_j] > 0:
                    transition_matrix[i,j] = pi[node_i,node_j]
            tmp_sum = transition_matrix[i,:].sum()
            if tmp_sum > 0:
                transition_matrix[i,:] /= tmp_sum
    
    return transition_matrix

#%%
@njit
def njit_all_RW_entropy_theoretic(do_ARW_step,start, types_entropy, indexes_to_save_M,
                                  pi,N,M,A,dt,alpha,k_list,s_fin,explored_links_matrix,r,rho,e_128,num_to_save,indexes_to_save,t_list,
                                  time_explored_nodes_list_N,time_explored_links_list_M,
                                  maximum_entr_prod_explored_list_time,empirical_entr_prod_list_time,
                                  maximum_entr_prod_explored_list_M,empirical_entr_prod_list_M,
                                  num_explored_links_list_time,num_explored_nodes_list_time, stop_at_covering_links = True):
    '''
        The transition matrix on which to calculate the entropy production rate is obtained using the corresponding values 
            from the original transition matrix based on type_entropy, and then renormalizing the rows.
        - type_entropy = 0 ---> entropy calculated on graph of explored nodes and all their links, 
            included first neighbors and their links to the explored nodes - NOT CALCULATED
        - type_entropy = 1 ---> entropy calculated on graph of explored nodes and all their links, 
            included first neighbors but considered different (unknown) for each link
            (multiple instances of the same neighbor for each of the links to the explored nodes) - NOT CALCULATED
        - type_entropy = 2 ---> entropy calculated on subgraph of explored nodes, with links only between themselves
        - type_entropy = 3 ---> entropy calculated on the graph of only explored links
        
        Input types
            - do_ARW_step: boolean
            - start: integer
            - indexes_to_save_M: 1D array of integer
            - pi: l1 normalized by rows 2D array of float, N rows x N columns
            - N: integer
            - M: integer
            - A: 2D array of integer, N rows x N columns
            - dt: integer
            - alpha: float
            - k_list: 1D array of integer
            - s_fin: float
            - explored_links_matrix: 2D array of float, N rows x N columns
            - r: 1D array of float
            - rho: 1D array of float
            - e_128: float
            - num_to_save: integer 
            - indexes_to_save: 1D array of integer
            - t_list: 1D array of integer
            - time_explored_nodes_list_N: 1D array of integer
            - time_explored_links_list_M: 1D array of integer
            - expected_entr_prod_explored_list_time: 2D array of float, 4 rows
            - empirical_entr_prod_list_time: 2D array of float, 4 rows
            - expected_entr_prod_explored_list_M: 2D array of float, 4 rows
            - empirical_entr_prod_list_M: 2D array of float, 4 rows
            - num_explored_links_list_time: 1D array of integer
            - num_explored_nodes_list_time: 1D array of integer
            - stop_at_covering_links: boolean
    '''
    # -1 is used to indicate not computed/not stored elements
    i_0 = np.random.randint(N) # inizializing node with highest occupancy to a random value
    maxIt = dt # max number of iterations
    next = -1 # next node, still not decided
    graph_explored = False # true if all the links are explored
    covering_time_nodes = -1 # if all the nodes are explored, here is stored the iteration when the last node is explored
    covering_time_links = -1 # if all the links are explored, here is stored the iteration when the last link is explored
    explored_nodes_list = [start] # list of all the different nodes explored so far
    num_explored_nodes = 1 # current number of different nodes explored in the process
    num_explored_links = 0 # current number of different links explored in the process
    index_M = 0 # index in indexes_to_save_M of next number of links where to calculate and store the entropy
    time_explored_nodes_list_N[num_explored_nodes-1] = 0 # at the beginning the first node is explored
    
    X_list = - np.ones(len(indexes_to_save)) # X_list is initialised
    index_saving = 0 # index in t_list of next iteration where to calculate and store the entropy
    a = 1. # initial annealing schedule parameter
    for it in range(maxIt):
        # Stationary distribution
        rho[start] += 1 # rho is not normalized here. If needed, divide by sum or (it+1)
        i_0 = np.where(rho == max(rho))[0][0] # update maximum occupancy index
        
        new_link = False # flag of new_link

        if it != 0:
            # update explored links and nodes, and calculate entropy
            if graph_explored == False and explored_links_matrix[start,next] == 0:
                # we have a new link start -- next
                new_link = True
                if num_explored_nodes != N and next not in explored_nodes_list:
                    # we have a new node next
                    explored_nodes_list.append(next)
                    num_explored_nodes += 1
                    time_explored_nodes_list_N[num_explored_nodes - 1] = it
                    if num_explored_nodes == N:
                        # all nodes have been explored
                        covering_time_nodes = it
                
                if num_explored_links + 1 == indexes_to_save_M[index_M]:
                    # calculate and store entropy and max entropy
                    for type_entropy in types_entropy:
                        transition_matrix = get_explored_transition_matrix(pi,explored_nodes_list,explored_links_matrix,type_entropy)
                        empirical_entr_prod_list_M[type_entropy,index_M], maximum_entr_prod_explored_list_M[type_entropy,index_M] = get_entropy(transition_matrix,it)
                    index_M +=1
                # update exploration variables
                explored_links_matrix[start,next] = 1
                explored_links_matrix[next,start] = 1
                num_explored_links += 1
                time_explored_links_list_M[num_explored_links - 1] = it
            if graph_explored == False and num_explored_links == M:
                # all the links are explored!
                graph_explored = True
                covering_time_links = it
                if stop_at_covering_links == True:
                    # we can exit the loop
                    break
            
            if num_to_save < 1 or (num_to_save > 1 and it == indexes_to_save[index_saving]):
                # compute analysis and store in the iteration-related arrays
                num_explored_links_list_time[index_saving] = num_explored_links
                num_explored_nodes_list_time[index_saving] = num_explored_nodes
                for type_entropy in types_entropy:
                    transition_matrix = get_explored_transition_matrix(pi,explored_nodes_list,explored_links_matrix,type_entropy)
                    empirical_entr_prod_list_time[type_entropy,index_saving], maximum_entr_prod_explored_list_time[type_entropy,index_saving] = get_entropy(transition_matrix,it)
                X_list[index_saving] = start
                t_list[index_saving] = it
                index_saving += 1
        
            # Finished to do dynamic analysis, starting now the update of the RW process
            start = next
            if do_ARW_step: # to be done only with ARW
                # step 0: setting the learning rate
                a = 1. / ((1 + it)**alpha)
                # calculate sum of elements in r related to the neighbors of start
                sum_tmp = 0.
                for jj in range(N):
                    if A[start,jj] == 1:
                        sum_tmp +=r[jj]
                for j in range(N):
                    if A[start,j] == 1:
                        # step 1: driven process update
                        pi[start,j] = r[j] / sum_tmp
                    
        # step 2: pick a random node according to the probability vector pi[start,:]
        # this is equivalent to do a random.choice, which is not supported with njit
        random_number = np.random.random()
        sum_tmp = 0.
        for jj in range(N):
            if A[start,jj] == 1:
                sum_tmp += pi[start,jj]
                if sum_tmp > random_number:
                    break
        next = jj

        if do_ARW_step: # to be done only with ARW
            # step 3: asynchronous update of the right eigenvector
            # calculate sum of elements in r related to the neighbors of start
            sum_tmp = 0.
            for jj in range(N):
                if A[start,jj] == 1:
                    sum_tmp += r[jj]
            r[start] = r[start] + a * ( e_128**( s_fin*np.log(k_list[start]) ) * sum_tmp / (k_list[start]*r[i_0]) - r[start] )

    # end of loop
    
    return covering_time_nodes,covering_time_links,num_explored_links_list_time,num_explored_nodes_list_time, \
           empirical_entr_prod_list_time,maximum_entr_prod_explored_list_time, \
           time_explored_nodes_list_N,time_explored_links_list_M,X_list,index_saving, \
           empirical_entr_prod_list_M, maximum_entr_prod_explored_list_M

def all_RW_preprocessing_entropy_theoretic(algorithm,G,s_fin=1,alpha=.1,dt=int(1e4), start = 0, num_to_save = 1000, renormalize_r = True,
                                           types_entropy = [2,3], stop_at_covering_links = True, saving=False, output_dir = "./data/", name_graph="graph"):

    '''
        This function prepares everything needed to run and analyse a random walker.

        Currently, the only supported algorithms are:
            - "URW" --> unbiased random walk
            - "MERW" --> maximum entropy random walk
            - "ARW" --> adaptive random walk, with learning rate defined by alpha and only run with the specified s_fin, no intermediate s.
                        The optimal tilting parameter is s_fin = 1, for which the process uses only local information to spread on a graph like a MERW on the subgraph explored.

        In this version, the empirical and maximum entropies are calculated when the time iteration or the number of links explored is one of the indexes chosen to save, according to num_to_save.
        These entropies are calculated on the subgraph formed by the explored nodes and all the links between them (row 2) and the subgraph containing only the links explored (row 3).
        The empirical entropies are calculated using the left eigenvector of the max eigenvalue of the current transition matrix, 
        while the maximum entropy is calculated as the logarithm of the maximum eigenvalue of the related adjacency matrix.

        Input
            - algorithm: name of the algorithm chosen. Supported "URW", "MERW", "ARW" (string)
            - G: graph. It has to be connected, otherwise the RW will not go from the initial connected component to any other one (nx.Graph())
            - s_fin: tilting parameter s on which the ARW runs, useful only if algorithm=="ARW" (float, default: 1)
            - alpha: annealing schedule exponent, useful only if algorithm=="ARW" (float, default: 0.1)
            - dt: number of iteration steps (integer, default: int(1e4))
            - start: starting node, suggested np.random.randint(N) (int, default: 0)
            - num_to_save: creates a geomspace and linspace between 1 and dt (or N, or M) to save the entropies (integer, default: 1000)
            - renormalize_r: if False, the initial array r in the ARW is a random array with values between 0 and 1;
                if True, it also gets renormalized to sum in l1 to 1 (boolean)
            - types_entropy: list of the types of entropy to calculate during the process. Give at least one of the following (list of integers, default: [2,3])
                - type_entropy = 0 ---> entropy calculated on graph of explored nodes and all their links, 
                    included first neighbors and their links to the explored nodes
                - type_entropy = 1 ---> entropy calculated on graph of explored nodes and all their links, 
                    included first neighbors but considered different (unknown) for each link 
                    (multiple instances of the same neighbor for each of the links to the explored nodes)
                - type_entropy = 2 ---> entropy calculated on subgraph of explored nodes, with links only between themselves
                - type_entropy = 3 ---> entropy calculated on the graph of only explored links
            - stop_at_covering_links: if true, it makes the simulation stop if the all links have been explored (boolean, default: True)
            - saving: saves the same dictionary that is returned into a file in a nested subdirectory of output_dir, with name_file automatically obtained to avoid overwriting (default: False)
            - output_dir: main directory where to create the subfolders where the results get stored (string, valid path)
            - name_graph: this is needed for the name of the saved file to identify the type of graph used in the simulation (string, default: "graph")
            
        Output 
            - to_save: dictionary with the following keys.
                - algorithm: algorithm chosen (string)
                - N: number of nodes in the graph (integer)
                - M: number of links in the graph (integer)
                - avg_k: average degree in the graph (float)
                - renormalize_r: whether or not the initial array r was renormalized (boolean)
                - dt: number of maximum iterations (integer)
                - start: starting node (integer)
                - alpha: annealing schedule parameter chosen (float)
                - s_fin: tilting parameter used in the process (float)
                - num_to_save: number of points used in the geomspace and linspace to save entropies (integer)
                - index_to_cut_time: last_index of t_list calculated, meaning that after that the simulation was stopped (integer)
                - t_list: list of all the iterations chosen where the entropies were calculated and stored (1D array of integer)
                - N_list: list of all the number of nodes chosen where the entropies were calculated and stored (1D array of integer)
                - M_list: list of all the number of links chosen where the entropies were calculated and stored (1D array of integer)
                - num_explored_nodes_list_time: list where the number of nodes at the iterations in t_list were stored (1D array of integer)
                - covering_time_nodes: iteration on which all the nodes got explored, -1 if this is not reached during the simulation (integer)
                - num_explored_links_list_time: list where the number of links at the iterations in t_list were stored (1D array of integer)
                - covering_time_links: iteration on which all the links got explored, -1 if this is not reached during the simulation (integer)
                - time_explored_nodes_list_N: list of iterations on which the process explored the corresponding number of nodes in N_list (1D array of integer)
                - time_explored_links_list_M: list of iterations on which the process explored the corresponding number of links in M_list (1D array of integer)
                - maximum_entr_prod: maximum entropy production rate of the whole graph (float)
                - empirical_entr_prod_list_time: empirical entropy production rate of the explored graph according to the different types of graph considered, 
                    calculated on the iterations in t_list (2D array of float, 4 rows)
                - maximum_entr_prod_explored_list_time: maximum entropy production rate of the explored graph according to the different types of graph considered, 
                    calculated on the iterations in t_list (2D array of float, 4 rows)
                - empirical_entr_prod_list_M: empirical entropy production rate of the explored graph according to the different types of graph considered, 
                    calculated when the number of links explored is in M_list (2D array of float, 4 rows)
                - maximum_entr_prod_explored_list_M: maximum entropy production rate of the explored graph according to the different types of graph considered, 
                    calculated when the number of links explored is in M_list (2D array of float, 4 rows)
                - X_list: list of nodes explored at the iterations in t_list (1D array of integer)
                - stop_at_covering_links: true or false whether the simulation was supposed to stop when all the links have been covered (boolean)
                - computational_time: computational time (calculated as end_time - start_time) needed in the process (datetime object)
    '''
    # Initialization
    start_time = datetime.datetime.now()
    # Get graph related parameters
    N = len(G.nodes()) # Size of the graph
    A = nx.adjacency_matrix(G).todense() # adjacency matrix
    M = len(G.edges()) # total number of links: PROBLEM IF ANTISYMMETRIC!?
    k_list = np.array(list(dict(G.degree()).values())) # degree list
    avg_k = np.mean(k_list)
    e = np.real(np.linalg.eigvals(A)) # eigenvalues
    max_eig = max(e) # maximum eigenvalue
    maximum_entr_prod=np.log(max_eig) # maximum entropy production rate
    
    # Get URW transition matrix (it's also how ARW starts)
    pi = normalize(A, axis=1,norm="l1") # transition matrix
    
    if "ARW" in algorithm:
        do_ARW_step = True
    else:
        do_ARW_step = False
        
    if algorithm == "MERW":
        # Get MERW transition matrix
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=600) #add max_iter if it does converge
        # Creation normalized eigenvector of max eigenvalue
        phi = list(eigenvector_centrality.values())
        phi2 = np.array(phi).reshape((1,N))
        phi2 = normalize(phi2,norm="l1")
        norm_max_eigenvector = phi2[0]
        # Creation MERW transition matrix
        S_tmp = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                S_tmp[i,j] = A[i,j]*norm_max_eigenvector[j] / max_eig / norm_max_eigenvector[i]
        # There is a very small error on how S_tmp is normalized, so let's normalize
        S = normalize(S_tmp, axis=1,norm="l1")
        pi = S
    
    
    # Other arrays needed in the function (some are needed only for ARW, but the function requires them anyway)
    r = np.array(np.random.random(size=N),dtype=np.float64) # in the ARW process this becomes the right eigenvector, that also updates pi. 
    if renormalize_r == True:
        r /= np.sum(r) # renormalization
    
    rho = np.zeros(N,dtype=np.float64) # occupancy vector
    
    # Create matrix of links explored
    explored_links_matrix = np.zeros((N,N), dtype=float) # Put 1 when element is explored
    e_128 = np.array([np.e],dtype=np.float64)[0] # calculates and gives the value of e with good precision. Unfortunately with njit float128 is not supported
    
    maxIt = dt # max number of iterations in the process
    
    # Get indexes where to save elements
    if num_to_save > 1:
        # indexes are found using the set of elements coming from a geomspace and linspace of length num_to_save
        indexes_to_save = np.array(sorted(list(set(np.geomspace(1,dt-1,num_to_save,dtype=int)).union(set(np.linspace(1,dt-1,num_to_save,dtype=int))))),dtype=int)
        indexes_to_save_M = np.array(sorted(list(set(np.geomspace(1,M,num_to_save,dtype=int)).union(set(np.linspace(1,M,num_to_save,dtype=int))))),dtype=int)
    else:
        # save all the indexes
        indexes_to_save = np.arange(dt,dtype=int)
        indexes_to_save_M = np.arange(M,dtype=int)
    
    # -1 is used to indicate not computed/not stored elements
    t_list = -np.ones(len(indexes_to_save),dtype=int) # store the iterations stored, to take from indexes_to_save
    
    maximum_entr_prod_explored_list_time = -np.ones((4,len(t_list)),dtype=float) # a row for each of the types of entropy to calculate
    empirical_entr_prod_list_time = -np.ones((4,len(t_list)),dtype=float) # a row for each of the types of entropy to calculate
    maximum_entr_prod_explored_list_M = -np.ones((4,len(indexes_to_save_M)),dtype=float) # a row for each of the types of entropy to calculate
    empirical_entr_prod_list_M = -np.ones((4,len(indexes_to_save_M)),dtype=float) # a row for each of the types of entropy to calculate
    
    time_explored_nodes_list_N = -np.ones(N,dtype=float)
    time_explored_links_list_M = -np.ones(M,dtype=float)
    
    num_explored_links_list_time = -np.ones(len(t_list),dtype=int)
    num_explored_nodes_list_time = -np.ones(len(t_list),dtype=int)
    
    # Call the process with all the initialised variables, and get the results
    covering_time_nodes,covering_time_links,num_explored_links_list_time,num_explored_nodes_list_time, \
            empirical_entr_prod_list_time,maximum_entr_prod_explored_list_time, \
            time_explored_nodes_list_N,time_explored_links_list_M,X_list,index_to_cut, \
            empirical_entr_prod_list_M, maximum_entr_prod_explored_list_M = njit_all_RW_entropy_theoretic( \
                     do_ARW_step,start, types_entropy, indexes_to_save_M, \
                     pi,N,M,A,dt,alpha,k_list,s_fin,explored_links_matrix,r,rho,e_128,num_to_save, \
                     indexes_to_save,t_list,time_explored_nodes_list_N,time_explored_links_list_M, \
                     maximum_entr_prod_explored_list_time,empirical_entr_prod_list_time, \
                     maximum_entr_prod_explored_list_M,empirical_entr_prod_list_M, \
                     num_explored_links_list_time,num_explored_nodes_list_time, stop_at_covering_links = stop_at_covering_links)
    
    end_time = datetime.datetime.now()
    print("Finished after %s"%(end_time-start_time), flush=True)
    
    # Store the results in a dictionary
    to_save =  {"algorithm":algorithm, "N":N, "M":M, "avg_k":avg_k, "renormalize_r":renormalize_r,\
                "dt":dt, "start":start, "alpha":alpha, "s_fin":s_fin, "num_to_save":num_to_save, \
                "index_to_cut_time":index_to_cut, "t_list":t_list, "M_list":indexes_to_save_M, \
                "num_explored_nodes_list_time":num_explored_nodes_list_time, "covering_time_nodes":covering_time_nodes, \
                "num_explored_links_list_time":num_explored_links_list_time, "covering_time_links":covering_time_links, \
                "time_explored_nodes_list_N":time_explored_nodes_list_N, "time_explored_links_list_M":time_explored_links_list_M, \
                "maximum_entr_prod":maximum_entr_prod, "empirical_entr_prod_list_time":empirical_entr_prod_list_time, \
                "maximum_entr_prod_explored_list_time":maximum_entr_prod_explored_list_time, \
                "empirical_entr_prod_list_M":empirical_entr_prod_list_M,"maximum_entr_prod_explored_list_M":maximum_entr_prod_explored_list_M, \
                "X_list":X_list, "stop_at_covering_links":stop_at_covering_links, \
                "computational_time":(end_time-start_time) \
               }
    
    if saving == True:
        # save the dictionary to_save in a file, which file_path has most of the information needed to identify the simulation run
        subfolder_path =  os.path.join(output_dir, name_graph, algorithm, f"dt_{dt}", f"N_{N}", "avg_k_%.2f"%(avg_k))
        if algorithm == "ARW":
            subfolder_path =  os.path.join(subfolder_path, "s_%.2f"%(s_fin), "alpha_%.2f"%(alpha))
        # create the subfolders if not present already
        os.makedirs(subfolder_path,exist_ok=True)
        # check if there are already files in the subfolder
        files_present = find_pattern("*.pkl",subfolder_path)
        # get the maximum ID present
        IDs_present = []
        for file in files_present:
            file = file.split("/")[-1]
            try:
                IDs_present.append(int(file[:file.index(".")]))
            except:
                pass
        max_ID = max(IDs_present) if len(IDs_present) > 0 else -1
        print("Found %d repetitions in the given folder with the same parameters."%(len(IDs_present)))
        if len(IDs_present) > 0:
            print("The maximum ID found is %d."%max_ID)
        # save this simulation with ID given by the maximum ID + 1
        ID_dump = max_ID + 1
        file_path = os.path.join(subfolder_path, f"{ID_dump}.pkl")
        print(f"Dumping to \n{file_path}")
        with open(file_path, "wb") as fp:
            joblib.dump(to_save, fp)
        end_time = datetime.datetime.now()
        print(f"Dumped results at {end_time}", flush = True)
        
        
        
        
#         os.makedirs("Results",exist_ok=True)
#         file_name = f"{name_graph}_{algorithm}_N_{N}_k_{np.mean(k_list)}_dt_{dt}_start_{start}_num_to_save_{num_to_save}"
#         if "ARW" in algorithm:
#             file_name += f"_s_fin_{s_fin}_alpha_{alpha}"
#         if stop_at_covering_links == True:
#             file_name += f"_stopped_at_covering_links"
#         print("Dumping to \n%s"%(os.path.join(os.getcwd(),f"Results/{file_name}.pkl")))
#         with open(os.path.join(f"Results/{file_name}.pkl"), 'wb') as fp:
#             joblib.dump(to_save,fp)
#         end_time = datetime.datetime.now()
#         print("Final results dumped after %s"%(end_time-start_time), flush=True)
        
    return to_save




def do_multiple_parameters_simulation(ID,graphs, algorithms, alpha, chosen_dts, num_to_save, types_entropy, stop_at_covering_links, output_dir, renormalize_r):
    '''
        Runs a simulation based on the parameters given and the ID.
        
        Input
            - ID: ID of the simulation, from which parameters are chosen (integer, starting from 0)
            - graphs: dict with correct name_graph as key, with value a dictionary with the keys N_requested, avg_k_requested, graph_dir, and max_tries (dict of dict)
            - algorithms: list of the keywords for the algorithms to run (list of string, supported ones are "URW", "MERW", "ARW")
            - alpha: annealing schedule parameter, useful only for ARW (float)
            - chosen_dts: list of all the dt to run the simulation for (list of integer)
            - num_to_save: number of elements in a geomspace and linspace on which to save the analysis on the entropy (integer)
            - types_entropy: list of the types of entropy to calculate during the process. Give at least one of the following (list of integers, default: [2,3])
                - type_entropy = 0 ---> entropy calculated on graph of explored nodes and all their links, 
                    included first neighbors and their links to the explored nodes
                - type_entropy = 1 ---> entropy calculated on graph of explored nodes and all their links, 
                    included first neighbors but considered different (unknown) for each link 
                    (multiple instances of the same neighbor for each of the links to the explored nodes)
                - type_entropy = 2 ---> entropy calculated on subgraph of explored nodes, with links only between themselves
                - type_entropy = 3 ---> entropy calculated on the graph of only explored links
            - stop_at_covering_links: if true, it makes the simulation stop if the all links have been explored (boolean)
            - output_dir: main directory where to create the subfolders where the results get stored (string, valid path)
            - renormalize_r: if False, the initial array r in the ARW is a random array with values between 0 and 1;
                if True, it also gets renormalized to sum in l1 to 1 (boolean)
    '''
    print("Starting ID %d"%(ID),flush=True)
    
    name_graph = list(graphs.keys())[ID % len(graphs)]
    algorithm = algorithms[(ID // len(graphs)) % len(algorithms)]
    dt = chosen_dts[(ID // len(graphs) // len(algorithms)) % len(chosen_dts)]
    repetition = ID // len(graphs) // len(algorithms) // len(chosen_dts)
    
    print(f"Simulation on graph {name_graph} with algorithm {algorithm}, using alpha={alpha} and dt ={dt}",flush=True)
    
    N_requested = graphs[name_graph]["N_requested"]
    avg_k_requested = graphs[name_graph]["avg_k_requested"]
    max_tries = graphs[name_graph]["max_tries"]
    graph_dir = graphs[name_graph]["graph_dir"]
    G, A = load_graph(name_graph = name_graph, N_requested = N_requested, avg_k_requested = avg_k_requested,  
                      graph_dir = graph_dir, max_tries = max_tries, show_graph = False, save_fig = False)

    largest_cc = max(nx.connected_components(G), key=len)
    print("Num nodes = %d"%(len(list(G.nodes()))))
    print("Len largest cc =",len(largest_cc))
    print("Num edges = %d"%(len(list(G.edges()))))
    print("Average degree =",np.mean(list(dict(G.degree()).values())))
    
    N = len(G.nodes()) # Size of the graph
    k_list = G.degree() # degree list
    A = nx.adjacency_matrix(G).todense() # adjacency matrix
    avg_k = np.mean(list(dict(G.degree()).values()))
    
    # getting a random starting node
    start = np.random.randint(N)
    print(f"Starting from {start}", flush = True)
    
    # Doing some calculations before starting, on the eigenvalues and hence the maximum entropies
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=600)
    eig_cent_list = np.array(list(eigenvector_centrality.values()))
    max_eig_node = np.real(np.argmax(eig_cent_list))
    median_eig_node = np.argmin(np.abs(eig_cent_list-np.median(eig_cent_list)))
    min_eig_node = np.argmin(eig_cent_list)
    e = np.real(np.linalg.eigvals(A)) # eigenvalues
    max_eig = max(e) # maximum eigenvalue
    max_entr_prod_rate = np.log(max_eig) # maximum eigenvalue
    print(f"Maximum entropy production rate is {max_entr_prod_rate}", flush = True)
    print(f"Node with maximum / median / minimum eigenvector centraility is {max_eig_node} / {median_eig_node} / {min_eig_node}", flush = True)

    # DOING ALL THE SIMULATIONS... CHOOSE BASED ON ID
    end_time = datetime.datetime.now()
    print(f"Finished preliminary calc. at {end_time}. Starting algorithm...", flush = True)


    to_save = all_RW_preprocessing_entropy_theoretic(algorithm, G, s_fin = 1, alpha = alpha, dt=int(dt), start = start, 
                                                     num_to_save = num_to_save, renormalize_r = renormalize_r,
                                                     types_entropy = types_entropy, stop_at_covering_links = stop_at_covering_links, 
                                                     saving = True, output_dir = output_dir, name_graph = name_graph)

    
#     subfolder_path =  os.path.join(output_dir, name_graph, algorithm, f"dt_{dt}", f"N_{N_requested}", "avg_k_%.2f"%(avg_k_requested))
#     if algorithm == "ARW":
#         subfolder_path =  os.path.join(subfolder_path, "s_%.2f"%(s_fin), "alpha_%.2f"%(alpha))
#     os.makedirs(subfolder_path,exist_ok=True)
#     file_name = f"repetition_{repetition}"
#     file_path = os.path.join(subfolder_path, file_name + ".pkl")
#     print(f"Dumping to \n{file_path}")
#     with open(file_path, "wb") as fp:
#         joblib.dump(to_save, fp)
#     end_time = datetime.datetime.now()
#     print(f"Dumped results at {end_time}", flush = True)
    return None