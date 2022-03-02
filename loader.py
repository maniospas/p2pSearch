import os
import numpy as np
import networkx as nx
from data import network, ir


def load_graph(node_init, dataset="fb"):
    graph = nx.Graph()
    node_dict = dict()
    filepath = network.get_edgelist_path(dataset)
    if not os.path.exists(filepath):
        network.download(dataset, filepath)
    with open(filepath) as file:
        for line in file:
            nodes = line[:-1].split(";")
            for node in nodes:
                if node not in node_dict:
                    node_dict[node] = node_init(node)
            graph.add_edge(node_dict[nodes[0]], node_dict[nodes[1]])
            graph.add_edge(node_dict[nodes[1]], node_dict[nodes[0]])
    return graph


def load_query_results(dataset="glove"):
    filepath = ir.get_qrels_path(dataset)
    if not os.path.exists(filepath):
        ir.download(dataset)
    with open(filepath, "r", encoding="utf8") as f:
        results = dict()
        for line in f:
            que_id, doc_id, _ = line.strip().split("\t")
            results[que_id] = doc_id
    return results


def load_embeddings(dataset="glove", type="docs"):
    filepath = ir.get_embeddings_path(dataset, type)
    if not os.path.exists(filepath):
        ir.download(dataset)
    many_arrays = np.load(filepath)
    ids = many_arrays["ids"]
    embs = many_arrays["embs"]
    return {idx: emb for idx, emb in zip(ids, embs)}


def load_texts(dataset="glove", type="docs"):
    filepath = ir.get_texts_path(dataset, type)
    if not os.path.exists(filepath):
        ir.download(dataset)
    with open(filepath, "r", encoding="utf8") as f:
        texts = dict()
        for line in f:
            idx, text = line.strip().split("\t")
            texts[idx] = text
    return texts


from nodes.base import PPRNode

# texts = load_texts("sts_benchmark", type="queries")
embs = load_texts("glove", type="other")
