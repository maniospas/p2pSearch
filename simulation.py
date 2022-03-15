import random
import numpy as np
import networkx as nx

import loader
from datatypes import Document, MessageQuery
from typing import List
from utils import analytic_ppr


class DecentralizedSimulation:
    def __init__(self, graph, _graph_name=None):
        self.nodes = list(graph.nodes)
        self.edges = list(graph.edges())
        self._graph = graph # will NOT be reshuffled
        self._graph_name = _graph_name

    def sample_nodes(self, k):
        return random.choices(self.nodes, k=k)

    def scatter_docs(self, documents: List[Document]):
        for node, doc in zip(random.choices(self.nodes, k=len(documents)), documents):
            node.add_doc(doc)

    def scatter_queries(self, queries: List[MessageQuery]):
        for node, query in zip(random.choices(self.nodes, k=len(queries)), queries):
            node.add_query(query)

    def run_embeddings(self, epochs, monitor=None):
        for time in range(epochs):
            print(f"EPOCH {time+1}")
            random.shuffle(self.edges)
            for u, v in self.edges:
                if random.random() < 0.5:
                    v.receive_embedding(u, u.send_embedding())
                    u.receive_embedding(v, v.send_embedding())

            if monitor is not None and not monitor():
                break

    def learn_neighbors(self):
        # initialize neighbors because they will not learn them otherwise
        for u, v in self.edges:
            u.neighbors[v] = v.embedding
            v.neighbors[u] = u.embedding

    def quick_embeddings(self, ppr_mat=None):
        if ppr_mat is None:
            adj = nx.adjacency_matrix(self._graph)
            nodes = list(self._graph.nodes)
            alpha = nodes[0].__class__.ppr_a
            ppr_mat = loader.analytic_ppr(adj, alpha, True, self._graph_name)
        nodes = self._graph.nodes
        personalizations = np.array([node.personalization for node in nodes])
        if personalizations.ndim > 2:
            embeddings = np.zeros_like(personalizations)
            for i in range(personalizations.shape[1]):
                embeddings[:, i, :] = ppr_mat @ personalizations[:, i, :]
        else:
            embeddings = ppr_mat @ personalizations

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        self.learn_neighbors()

    def run_queries(self, epochs, monitor):
        nodes_to_check = self.nodes  # TODO remove perf shortcut
        for time in range(epochs):
            # random.shuffle(self.nodes)  # TODO remove perf shortcut
            outgoing = {}
            for u in nodes_to_check:
                if u.has_queries_to_send():  # and random.random() < 0.8: # the probabilities does not matter
                    outgoing[u] = u.send_queries()

            nodes_to_check = [] # TODO remove perf shortcut
            for u, to_send in outgoing.items():
                for v, queries in to_send.items():
                    v.receive_queries(queries, u)
                    nodes_to_check.append(v)

            if len(nodes_to_check) == 0:
                break

            if monitor is not None and not monitor():
                break
        return time

    def __call__(self, epochs, monitor=None):
        time = 0
        for time in range(epochs):
            random.shuffle(self.edges)
            for u, v in self.edges:
                if random.random() < 0.1:
                    mesg_to_v, mesg_to_u = u.send_embedding(), v.send_embedding()
                    v.receive_embedding(u, mesg_to_v)
                    u.receive_embedding(v, mesg_to_u)

            random.shuffle(self.nodes)
            outgoing = {}
            for u in self.nodes:
                if u.has_queries_to_send() and random.random() < 1.0:
                    outgoing[u] = u.send_queries()
            for u, to_send in outgoing.items():
                for v, queries in to_send.items():
                    v.receive_queries(queries, u)

            if monitor is not None and not monitor():
                break
        return time

    # TODO: dangerous
    def clear(self):
        for node in self.nodes:
            node.clear()