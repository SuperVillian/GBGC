import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from scipy.linalg import eigh
import networkx as nx
import parse
import classification
import netlsd
import time
from collections import deque

def sym_normalize_adj(adj):
    deg = adj.sum(1)
    deg_inv = np.where(deg > 0, 1. / np.sqrt(deg), 0)
    return np.einsum('i,ij,j->ij', deg_inv, adj, deg_inv)

def normalizeLaplacian(G):
    n = G.shape[0]
    return np.eye(n) - sym_normalize_adj(G)

def assign_nodes_to_multiple_centers(G, centers):
    # Initialize a dictionary to store the set of nodes belonging to each center
    center_nodes_dict = {center: set() for center in centers}

    # Initialize queues and a visited record
    queues = {center: deque([center]) for center in centers}
    visited = {center: center for center in centers}

    # Perform multi-source BFS
    while any(queues.values()):
        for center in centers:
            if queues[center]:
                current_node = queues[center].popleft()
                center_nodes_dict[center].add(current_node)

                for neighbor in G.neighbors(current_node):
                    if neighbor not in visited:
                        visited[neighbor] = center
                        queues[center].append(neighbor)

    # Return the result
    return center_nodes_dict


def qity(graph):
    avg_degree = graph.number_of_edges() / len(graph)
    global_clustering = nx.transitivity(graph)
    qity =avg_degree + global_clustering
    return qity


# Evaluate the most suitable value for Granular-Ball division and select the largest division
def split_metric(father_qity, son_qity):
    metric = (son_qity-father_qity)/father_qity
    return son_qity


def split_ball(GB_info):
    # GB_info = [GB,father_qity,son_qity,split_metric]
    GB_graph = GB_info[0]
    degree_dict = dict(GB_graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    center_nodes = sorted_nodes[:2]
    center_nodes_dict = assign_nodes_to_multiple_centers(GB_graph, center_nodes)
    clusters = [cluster for cluster in center_nodes_dict.values()]
    cluster_a = clusters[0]  
    cluster_b = clusters[1]
        
    
    GB_graph_a = nx.subgraph(GB_graph, cluster_a)
    GB_graph_b = nx.subgraph(GB_graph, cluster_b)
    
    father_qity = qity(GB_graph)
    son_qity_a = qity(GB_graph_a)
    son_qity_b = qity(GB_graph_b)

    son_info_a = [GB_graph_a, father_qity, son_qity_a, 0]
    son_info_b = [GB_graph_b, father_qity, son_qity_b, 0]

    return son_info_a, son_info_b  

def split_graph(graph, init_GB_num):
    sqrt_n = init_GB_num
    
    # Get the node with the highest degree
    max_degree_node = max(graph.degree, key=lambda x: x[1])[0]
    
    # Create a queue for BFS
    queue = deque([max_degree_node])
    visited = set()
    subgraph_nodes = set([max_degree_node])
    visited.add(max_degree_node)

    while queue:
        # Get the number of nodes in the current level
        current_level_size = len(queue)
        current_layer_nodes = []
        
        # Traverse all nodes in the current level
        for _ in range(current_level_size):
            node = queue.popleft()
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    current_layer_nodes.append(neighbor)
        
        # After traversing a level, check if the subgraph node count exceeds sqrt_n
        if len(subgraph_nodes) + len(current_layer_nodes) > sqrt_n:
            break
        
        # Add the current level nodes to the subgraph and queue
        subgraph_nodes.update(current_layer_nodes)
        queue.extend(current_layer_nodes)
    
    # Remove selected nodes from the original graph
    remaining_graph = graph.copy()
    remaining_graph.remove_nodes_from(subgraph_nodes)
    
    # Return the first subgraph and the remaining graph
    return max_degree_node, remaining_graph

def init_GB_graph(graph, init_GB_num, n):

    if n < init_GB_num:
        init_GB_list = [[graph]]
        father_qity = qity(graph)
        son_qity = qity(graph)
        init_GB_list[0].append(father_qity)
        init_GB_list[0].append(son_qity)
        init_GB_list[0].append(0)
        return init_GB_list
    
    remaining_graph = graph
    center_nodes = []
    for i in range(init_GB_num):
        max_degree_node, remaining_graph = split_graph(remaining_graph, init_GB_num)
        center_nodes.append(max_degree_node)
    center_nodes_dict = assign_nodes_to_multiple_centers(graph, center_nodes)
    init_GB_list = [[nx.subgraph(graph, cluster)] for cluster in center_nodes_dict.values()]
    father_qity = qity(graph)
    for init_GB in init_GB_list:
        init_GB.append(father_qity)
        son_qity = qity(init_GB[0])
        init_GB.append(son_qity)
        init_GB.append(split_metric(father_qity, son_qity))
    return init_GB_list

# [GB_graph_a, father_qity, son_qity_a, 0]

def sort_list(lst):
    # Using the sorted function to obtain sorted values
    sorted_values = sorted(lst)

    # Get the position of each value in the sorted list
    sorted_positions = [sorted_values.index(value) for value in lst]
    return sorted_positions

def update(GB_list):
    # print("len(GB_list)",len(GB_list))
    num_list = []
    qity_list = []
    for GB in GB_list:
        num_list.append(len(GB[0]))
        qity_list.append(split_metric(GB[1], GB[2]))
    num_list = sort_list(num_list)
    qity_list = sort_list(qity_list)
    # print("num_list",num_list)
    # print("qity_list",qity_list)
    for i in range(len(GB_list)):
        GB_list[i][3] = num_list[i] + qity_list[i]
    
    return GB_list



def get_GB_graph(graph, r, init_methods="two"):
    n = int(np.ceil(r*len(graph)))
    if init_methods == "two":
        init_GB_num = 2
    else:
        import math
        init_GB_num = math.isqrt(len(graph))
        # init_GB_num =  int(math.log2(len(graph)))
    

    GB_list = init_GB_graph(graph, init_GB_num, n)
    # init_GB_list = [[init_GB,father_qity,son_qity,split_metric], [init_GB,father_qity,son_qity,split_metric]]
    # print("GB_list",GB_list)
    while len(GB_list) < n:
        GB_list = update(GB_list)
        index_of_max = max(enumerate(GB_list), key=lambda x: x[1][3])[0]
        son_info_a, son_info_b = split_ball(GB_list[index_of_max])
        GB_list.pop(index_of_max)
        GB_list.append(son_info_a)
        GB_list.append(son_info_b)
    GB_graph = nx.Graph()
    if len(GB_list) == 1:
        return graph
    for i in range(len(GB_list)):
        for j in range(i+1, len(GB_list)):
            flag = False
            count = 0
            for a in GB_list[i][0].nodes():
                for b in GB_list[j][0].nodes():
                    if graph.has_edge(a, b):
                        count += 1
                        flag = True
            if flag:
                GB_graph.add_edge(i, j, weight = count)
    return GB_graph

dir = 'dataset'
datasets = ["MUTAG", "PROTEINS",  "IMDB-BINARY", "NCI109","DHFR","BZR","Tox21_AR-LBD_testing","OVCAR-8H","P388H","SF-295H"]
# datasets = ["MUTAG"]
for r in np.arange(0.1, 1, 0.1):
    print("Coarsening rate:", r)
    for dataset in datasets:
        am, labels = parse.parse_dataset(dir, dataset)
        G_list = []
        # am: adjacency matrix
        num_samples = len(am) # graph number
        # exit()

        # graph coarsening
        current_timestamp = time.time()
        for i in range(num_samples):
            N = am[i].shape[0]
            graph = nx.from_numpy_matrix(am[i])
            init_methods = "radical n"
            # To determine if the graph is connected, if it is not connected, divide the graph into multiple connected subgraphs
            if nx.is_connected(graph):
                GB_graph = get_GB_graph(graph, r, init_methods)
            else:
                connected_components = list(nx.connected_components(graph))
                connected_subgraphs = [graph.subgraph(component) for component in connected_components]
                GB_graph_list = []
                for connected_subgraph in connected_subgraphs:
                    GB_graph = get_GB_graph(connected_subgraph, r, init_methods)
                    GB_graph_list.append(GB_graph)
                merged_graph = nx.Graph()
                for GB_graph in GB_graph_list:
                    merged_graph = nx.compose(merged_graph, GB_graph)
                GB_graph = merged_graph
            # print(i,"Original number of nodes:",N," Number of Granular-Ball:", len(GB_graph))
                    
            GB_graph_am = nx.to_numpy_array(GB_graph)
            # print(len(GB_graph), len(graph))
            G = eigh(normalizeLaplacian(GB_graph_am), eigvals_only=True)
            G_list.append(G)
        end_timestamp1 = time.time()
        times = end_timestamp1-current_timestamp
        print(dataset,"GBGC_time：",round(times, 2),"seconds")

        # train
        h = 250
        X = np.zeros((num_samples, h))
        Y = labels
        t = np.logspace(-2, 2, h)
        for i in range(len(G_list)):
            X[i] = netlsd.heat(G_list[i], t, normalization="empty")    
        acc, std = classification.KNN_classifier_nfold(X, Y, n=10, k=1)
        print(dataset, "acc={:.2f}\\pm{:.2f}".format(acc, std / np.sqrt(10)))
    print()
# nohup python -u GBGC_ratio.py > ./result/GBGC_ratio.log 2>&1 &
