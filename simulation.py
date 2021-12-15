import random
from datatypes import Document, ExchangedQuery
from typing import List


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


class SynchronousDecentralizedSimulation:
    def __init__(self, graph):
        self.nodes = list(graph.nodes)
        self.edges = list(graph.edges())
        self._graph = graph

        for u, v in self.edges:
            u.learn_neighbor(v)
            v.learn_neighbor(u)

    # def _get_neighbor_set(self, nodes):
    #     neighbors_set = set()
    #     for node in nodes:
    #         neighbors_set.update(set(self._graph.neighbors(node)))
    #     return neighbors_set

    def scatter_docs(self, documents: List[Document]):
        for node, doc in zip(random.choices(self.nodes, k=len(documents)), documents):
            node.add_doc(doc)
            node.update()

    def scatter_queries(self, queries):
        query_objects = list()
        for test_node, query in zip(random.choices(self.nodes, k=len(queries)), queries):
            query_objects.append(query)
            test_node.add_query(query)
        return query_objects

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

    def _advertise(self, init_node, radius=1):
        for hop, edges in enumerate(self.stream_hops(init_node, radius, return_edges=True), 1):
            for from_node, node in edges:
                node.learn_neighbor(from_node, init_node.embedding)

    def advertise(self, radius=1):
        for node in self.nodes:
            self._advertise(node, radius)

    def __call__(self, epochs, monitor=None):
        for time in range(epochs):
            to_forward = dict()
            for node in self.nodes:
                to_forward.update(node.forward())
            print(f"{time}: {to_forward}")
            for to_node, messages in to_forward.items():
                to_node.receive(node, messages)
            if monitor is not None and not monitor():
                break
        return time

