from datatypes import Text, TTLQuery
from collections import defaultdict
import random as rnd


class BaseSimulation:

    def __init__(self, *, graph, node2docs, node2queries, advertise_radius, ttl, capacity, epochs, init_node, monitor):
        self.graph = graph
        self.node2docs = node2docs
        self.node2queries = node2queries
        self.advertise_radius = advertise_radius
        self.ttl = ttl
        self.capacity = capacity
        self.epochs = epochs
        self.monitor = monitor
        self.init_node = init_node

    def create_document_nodes(self):
        return {u: self.init_node(u) for u in self.graph.nodes}

    def connect_document_nodes(self, docnodes):
        for u, v in self.graph.edges:
            docnodes[u].learn_neighbor(docnodes[v])
            docnodes[v].learn_neighbor(docnodes[u])

    def store_docs(self, docnodes):
        for u, docs in self.node2docs.items():
            for doc in docs:
                docnodes[u].add_doc(doc)

    def create_queries(self):
        ttl_queries = {}
        for queries in self.node2queries.values():
            for query in queries:
                ttl_query = TTLQuery(query.name, query.embedding, self.ttl, self.capacity)
                ttl_queries[query] = ttl_query
        return ttl_queries

    def store_queries(self, docnodes, ttlqueries):
        for u, queries in self.node2queries.items():
            for q in queries:
                docnodes[u].add_query(ttlqueries[q])

    def advertise_node(self, docnodes, init_u):
        for hop, edges in enumerate(self.stream_hops(init_u, max_hop=self.advertise_radius, return_edges=True), 1):
            # print(f"node {init_node} hop {hop}")
            for u, v in edges:
                docnodes[v].learn_neighbor(docnodes[u], docnodes[init_u].embedding)  # must be customizable

    def advertise(self, docnodes):
        for u in self.graph.nodes:
            if len(docnodes[u].docs) > 0:
                self.advertise_node(docnodes, u)
        # print(f"{self} finished advertising")

    def stream_hops(self, init_node, max_hop=float('inf'), return_edges=False):
        hop = 0
        current_nodes, visited_nodes, next_nodes = {init_node}, {init_node}, set(self.graph.neighbors(init_node))
        edges = [(init_node, neighbor) for neighbor in next_nodes]
        while len(next_nodes) > 0 and hop < max_hop:
            hop += 1
            yield edges if return_edges else list(next_nodes)

            current_nodes = next_nodes
            visited_nodes.update(current_nodes)
            next_nodes = set()
            edges = []
            for node in current_nodes:
                next_nodes_per_node = set(self.graph.neighbors(node))
                next_nodes_per_node.difference_update(visited_nodes)
                edges.extend([(node, neighbor) for neighbor in next_nodes_per_node])
                next_nodes.update(next_nodes_per_node)

    def __call__(self):

        docnodes = self.create_document_nodes()
        self.connect_document_nodes(docnodes)
        self.store_docs(docnodes)
        self.advertise(docnodes)

        ttlqueries = self.create_queries()
        self.store_queries(docnodes, ttlqueries)
        results = ttlqueries.values()

        # run
        for time in range(self.epochs):
            to_forward = []
            for docnode in docnodes.values():
                for receiver, messages in docnode.forward().items():
                    to_forward.append((docnode, receiver, messages))

            for sender, receiver, messages in to_forward:
                receiver.receive(sender, messages)

            if self.monitor is not None and not self.monitor(time, results):
                break

        return time, results


# class OriginSimulation:
#
#     def __init__(self, *, graph, docs, query, advertise_radius, ttl, capacity, epochs, init_node, monitor):
#
#         node2docs = defaultdict(lambda: [])
#         for node, doc in zip(rnd.choices(list(graph.nodes), k=len(docs)), docs):
#             node2docs[node].append(doc)
#
#         queries = [Text(f"{query.name} clone {i + 1}", query.embedding) for i in range(len(graph.nodes))]
#         node2queries = defaultdict(lambda: [])
#         for node, query in zip(list(graph.nodes), queries):
#             node2docs[node].append(query)
#
#         self.sim = BaseSimulation(graph, node2docs, node2queries, advertise_radius, ttl, capacity, epochs,
#                                   init_node, monitor)
#
#     def __call__(self):
#         _, results = self.sim()
#         return [query.candidate_docs for query in results]
#
#
# class SingleQuerySingleOriginSimulator:
#
#     def __init__(self, *, graph, docs, query, n_query_reps,
#                  advertise_radius, ttl, capacity, epochs, init_node, monitor, origin=None):
#
#         node2docs = defaultdict(lambda: [])
#         for node, doc in zip(rnd.choices(list(graph.nodes), k=len(docs)), docs):
#             node2docs[node].append(doc)
#
#         self.origin = rnd.choice(list(graph.nodes)) if origin is None else origin
#         queries = [Text(f"{query.name} clone {i+1}", query.embedding) for i in range(n_query_reps)]
#         node2queries = defaultdict(lambda: [])
#         node2queries[self.origin].extend(queries)
#
#         self.sim = BaseSimulation(graph, node2docs, node2queries, advertise_radius, ttl, capacity, epochs,
#                                   init_node, monitor)
#
#     def __call__(self):
#         return self.sim()
#
#
# class SingleQueryMultipleOriginSimulator:
#
#     def __init__(self, *, graph, docs, query, n_query_reps,
#                  advertise_radius, ttl, capacity, epochs, monitor):
#
#         node2docs = defaultdict(lambda: [])
#         for node, doc in zip(rnd.choices(list(graph.nodes), k=len(docs)), docs):
#             node2docs[node].append(doc)
#
#         queries = [Text(f"{query.name} - clone {i+1}", query.embedding) for i in range(n_query_reps)]
#         node2queries = defaultdict(lambda: [])
#         for node, query in zip(rnd.choices(list(graph.nodes), k=len(queries)), queries):
#             node2queries[node].append(query)
#
#         self.sim = BaseSimulation(graph, node2docs, node2queries, advertise_radius, ttl, capacity, epochs, monitor)
#
#     def __call__(self):
#         return self.sim()
