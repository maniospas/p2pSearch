import networkx as nx
import os
from loader import BASEDIR

def mean_neighbors(graph):
    degs = [len(list(graph.neighbors(node))) for node in graph.nodes]
    return sum(degs) / len(degs)


def generate_erdos_graph(n_nodes, target_mean_neighbors):
    return nx.gnp_random_graph(n_nodes, target_mean_neighbors / (n_nodes-1))


def generate_albert_barabasi_graph(n_nodes, target_mean_neighbors):
    start = int(target_mean_neighbors / 2)
    stop = int(target_mean_neighbors)
    old_graph = graph = nx.barabasi_albert_graph(n_nodes, start)
    deg = [len(list(graph.neighbors(node))) for node in graph.nodes]
    old_mean_deg = mean_deg = sum(deg) / len(deg)
    for m in range(start+1, stop):
        graph = nx.barabasi_albert_graph(n_nodes, m)
        deg = [len(list(graph.neighbors(node))) for node in graph.nodes]
        mean_deg = sum(deg) / len(deg)
        if mean_deg > target_mean_neighbors:
            break
        old_graph = graph
        old_mean_deg = mean_deg

    if abs(old_mean_deg - target_mean_neighbors) < abs(mean_deg - target_mean_neighbors):
        return old_graph
    else:
        return graph


n_nodes = 4039
target_mean_neighbors = 43.69

dir_path = os.path.join(BASEDIR, "data")
delimiter = ";"
graphs = [generate_erdos_graph(n_nodes, target_mean_neighbors), generate_albert_barabasi_graph(n_nodes, target_mean_neighbors)]
labels = ["erdos", "albertbarabasi"]

for graph, label in zip(graphs, labels):

    print(f"graph {label}: {graph.number_of_nodes()} nodes, {mean_neighbors(graph)} mean neighbors")

    path = os.path.join(dir_path, f"{label}_undirected_edgelist.csv")

    with open(path, "w", encoding="utf8") as f:
        for edge in graph.edges:
            f.write(f"{edge[0]}{delimiter}{edge[1]}\n")





