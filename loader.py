import os
import numpy as np
import networkx as nx

import utils
from data import network, ir
from utils import analytic_ppr
from stubs import StubNode
from importlib import import_module
from subprocess import run


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


def load_node2vec(dataset, as_dict=False, normalized=False, params=None):
    params = params or dict()
    dim = params.get("dim", 768)
    l = params.get("l", 3)
    p = params.get("p", 1)
    q = params.get("q", 0.3)

    basedir = os.path.join(os.path.dirname(__file__), "data", "node2vec")
    emb_path = os.path.join(basedir, f"{dataset}_dim{dim}_l{l}_p{p}_q{q}.emb")
    if not os.path.exists(emb_path):
        edgelist_path = os.path.join(basedir, f"{dataset}.edgelist")
        if not os.path.exists(edgelist_path):
            raise Exception(f"no node2vec data for dataset {dataset}")

        script_path = os.path.join(basedir, "node2vec")
        args = f"{script_path} -i:{edgelist_path} -o:{emb_path} -l:{l} -d:{dim} -p:{p} -q:{q} -v".split()
        run(args)

    with open(emb_path, "r", encoding="utf8") as f:
        f.readline()
        ids = []
        embs = []
        for line in f:
            tokens = line.rstrip().split()
            ids.append(tokens[0])
            emb = np.array([float(token) for token in tokens[1:]])
            if normalized:
                emb = utils.unitary(emb)
            embs.append(emb)
    if as_dict:
        return {id_: emb for id_, emb in zip(ids, embs)}
    else:
        return ids, np.array(embs)


def ppr_matrix(dataset, alpha, symmetric=True):
    filepath = network.get_ppr_matrix_path(dataset, alpha, symmetric)
    if os.path.exists(filepath):
        return np.load(filepath)

    graph = load_graph(StubNode, dataset)
    ppr_matrix = utils.analytic_ppr(nx.adjacency_matrix(graph), alpha, symmetric)
    np.save(filepath, ppr_matrix)
    return ppr_matrix


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


def load_clusters(dataset, n_clusters):
    filepath = ir.get_clusters_path(dataset, n_clusters)
    if not os.path.exists(filepath):
        store_clusters = getattr(import_module('data.ir.glove.create_clusters'), 'store_clusters')
        store_clusters(n_clusters)
    return np.load(filepath)


def load_all(dataset):
    query_results = load_query_results()
    que_embs = load_embeddings(dataset=dataset, type="queries")
    doc_embs = load_embeddings(dataset=dataset, type="docs")
    other_doc_embs = load_embeddings(dataset=dataset, type="other_docs")
    dim = len(next(iter(other_doc_embs.values())))
    return dim, query_results, que_embs, doc_embs, other_doc_embs

