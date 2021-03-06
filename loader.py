import networkx as nx
import numpy as np


def load_graph(node_init, dataset="fb"):
    graph = nx.Graph()
    node_dict = dict()
    with open('data/'+dataset+"_undirected_edgelist.csv") as file:
        for line in file:
            nodes = line[:-1].split(";")
            for node in nodes:
                if node not in node_dict:
                    node_dict[node] = node_init(node)
            graph.add_edge(node_dict[nodes[0]], node_dict[nodes[1]])
            graph.add_edge(node_dict[nodes[1]], node_dict[nodes[0]])
    return graph


def load_query_results(dataset="antique_test"):
    results = dict()
    with open('data/'+dataset+"_qrels.csv") as file:
        for line in file:
            line = line[:-1].split(";")
            results[line[0]] = line[1]
    return results


def load_embeddings(dataset="antique_test", type="docs"):
    docs = dict()
    with open('data/'+dataset+"_"+type+".csv") as file:
        for line in file:
            line = line.split(";")
            docs[line[0]] = np.array(eval(line[1]))
    return docs