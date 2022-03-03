from loader import *
from simulation import DecentralizedSimulation
from nodes import *

import numpy as np
import random
import networkx as nx
import scipy.sparse as sparse
from scipy.sparse.linalg import inv


class EmbeddingMonitor:

    def __init__(self, nodes):
        self.nodes = nodes
        self.embs = [[node.embedding for node in nodes]]

    @property
    def embeddings(self):
        return np.array(self.embs)

    def monitor(self):
        self.embs.append([node.embedding for node in self.nodes])
        return True

def exact_ppa(adj, ppr_a, personalization):
    I = sparse.identity(adj.shape[0])
    invdiag = sparse.diags(np.array(adj.sum(axis=1)).squeeze() ** -0.5)
    return ppr_a * inv(I - (1 - ppr_a) * invdiag @ adj @ invdiag) @ personalization

random.seed(10)
n_docs = 20
epochs = 100
ppr_a = 0.1

# load data
other_doc_embs = load_embeddings(dataset="glove", type="other_docs")
dim = len(next(iter(other_doc_embs.values())))

node_init = lambda name: HardSumEmbeddingNode(name, ppr_a=ppr_a, init_personalization=lambda: np.zeros(dim))
graph = load_graph(node_init, "fb")
simulation = DecentralizedSimulation(graph)

docs = [Document(doc_id, other_doc_embs[doc_id]) for doc_id in random.sample(list(other_doc_embs), n_docs)]
simulation.scatter_docs(docs)

personalizations = np.array([node.personalization for node in simulation.nodes])
adj = nx.linalg.adjacency_matrix(graph)
personalizations = np.array([node.personalization for node in simulation.nodes])
ppr_embs = exact_ppa(adj, ppr_a=0.1, personalization=personalizations)


print("Warming up")
embmonitor = EmbeddingMonitor(simulation.nodes)
simulation.run_embeddings(epochs, embmonitor.monitor)

diffs = embmonitor.embeddings - ppr_embs
diffs = np.linalg.norm(diffs, axis=2, ord=1).max(axis=1)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(diffs)
