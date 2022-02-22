import networkx as nx
import numpy as np
import os

from dirs import DATA_DIR

def load_graph(node_init, dataset="fb"):
    graph = nx.Graph()
    node_dict = dict()
    filename = os.path.join(DATA_DIR, "network", dataset+"_undirected_edgelist.csv")
    with open(filename) as file:
        for line in file:
            nodes = line[:-1].split(";")
            for node in nodes:
                if node not in node_dict:
                    node_dict[node] = node_init(node)
            graph.add_edge(node_dict[nodes[0]], node_dict[nodes[1]])
            graph.add_edge(node_dict[nodes[1]], node_dict[nodes[0]])
    return graph


def load_query_results(dataset="glove"):
    filepath = os.path.join(DATA_DIR, dataset, "qrels.txt")
    with open(filepath, "r", encoding="utf8") as f:
        results = dict()
        for line in f:
            que_id, doc_id, _ = line.strip().split("\t")
            results[que_id] = doc_id
    return results


def load_embeddings(dataset="glove", type="docs"):
    filepath = os.path.join(DATA_DIR, dataset, type+"_embs.npz")
    many_arrays = np.load(filepath)
    ids = many_arrays["ids"]
    embs = many_arrays["embs"]
    return {idx: emb for idx, emb in zip(ids, embs)}


def load_texts(dataset="glove", type="docs"):
    filepath = os.path.join(DATA_DIR, dataset, type+".txt")
    with open(filepath, "r", encoding="utf8") as f:
        texts = dict()
        for line in f:
            idx, text = line.strip().split("\t")
            texts[idx] = text
    return texts

