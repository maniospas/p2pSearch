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


class RandomWalkWithAdvertisementsSimulation:
    def __init__(self, graph, documents: List[Document], advertise_radius=0):
        self.nodes = list(graph.nodes)
        self.edges = list(graph.edges())
        self._graph = graph
        self.documents = documents
        self.advertise_radius = advertise_radius

        self._learn_neighbors()
        self._scatter_docs(documents)
        self._advertise(advertise_radius)

    def _learn_neighbors(self):
        for u, v in self.edges:
            u.learn_neighbor(v)
            v.learn_neighbor(u)

    def _scatter_docs(self, documents: List[Document]):
        for node, doc in zip(random.choices(self.nodes, k=len(documents)), documents):
            node.add_doc(doc)

    def _advertise_node(self, init_node, radius=1):
        for hop, edges in enumerate(self.stream_hops(init_node, radius, return_edges=True), 1):
            print(f"node {init_node} hop {hop}")
            for from_node, node in edges:
                node.learn_neighbor(from_node, init_node.embedding)

    def _advertise(self, radius=1):
        for node in self.nodes:
            if len(node.docs) > 0:
                self._advertise_node(node, radius)
        print(f"{self} finished advertising")

    def shuffle_docs(self):
        for node in self.nodes:
            node.clear_docs()
        self._scatter_docs(self.documents)
        self._advertise(self.advertise_radius)

    def change_advertise_radius(self, radius):
        for node in self.nodes:
            node.clear_neighbors()
        self._advertise(radius)

    def sample_nodes(self, k):
        return rnd.sample(self.nodes, k)

    def stream_hops(self, init_node, max_hop=float('inf'), return_edges=False,):
        hop = 0
        current_nodes, visited_nodes, next_nodes = {init_node}, {init_node}, set(self._graph.neighbors(init_node))
        edges = [(init_node, neighbor) for neighbor in next_nodes]
        while len(next_nodes) > 0 and hop < max_hop:
            hop += 1
            yield edges if return_edges else list(next_nodes)

            current_nodes = next_nodes
            visited_nodes.update(current_nodes)
            next_nodes = set()
            edges = []
            for node in current_nodes:
                next_nodes_per_node = set(self._graph.neighbors(node))
                next_nodes_per_node.difference_update(visited_nodes)
                edges.extend([(node, neighbor) for neighbor in next_nodes_per_node])
                next_nodes.update(next_nodes_per_node)

    def clear_queries(self):
        for node in self.nodes:
            node.clear_queries()

    def __call__(self, epochs, nodes2queries, ttl, capacity, monitor=None):

        self.clear_queries()

        results = []
        for node, queries in nodes2queries.items():
            for query in queries:
                ttl_query = TTLQuery(query.name, query.embedding, ttl, capacity)
                node.add_query(ttl_query)
                results.append(ttl_query)

        for time in range(epochs):
            to_forward = defaultdict(lambda: [])
            for node in self.nodes:
                for to_node, messages in node.forward().items():
                    to_forward[to_node].extend(messages)

            for to_node, messages in to_forward.items():
                to_node.receive(node, messages)
            if monitor is not None and not monitor(time, results):
                break
        return time, results

