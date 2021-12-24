import random
from datatypes import Document, ExchangedQuery, TTLQuery
from typing import List
import random as rnd
from collections import defaultdict


class DecentralizedSimulation:
    def __init__(self, graph):
        self.nodes = list(graph.nodes)
        self.edges = list(graph.edges())

    def scatter_docs(self, documents: List[Document]):
        for node, doc in zip(random.sample(self.nodes, len(documents)), documents):
            node.add_doc(doc)
            node.update()

    def scatter_queries(self, queries: List[ExchangedQuery]):
        query_objects = list()
        for test_node, query in zip(random.sample(self.nodes, len(queries)), queries):
            query_objects.append(query)
            test_node.add_query(query)
        return query_objects

    def __call__(self, epochs, monitor=None):
        for time in range(epochs):
            random.shuffle(self.edges)
            for u, v in self.edges:
                if random.random() < 0.1:
                    v.receive(u, u.send(v))
                    u.receive(v, v.send(u))
            if monitor is not None and not monitor():
                break
        return time




