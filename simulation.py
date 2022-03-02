import random
from datatypes import Document, MessageQuery
from typing import List


class DecentralizedSimulation:
    def __init__(self, graph):
        self.nodes = list(graph.nodes)
        self.edges = list(graph.edges())

    def sample_nodes(self, k):
        return random.choices(self.nodes, k=k)

    def scatter_docs(self, documents: List[Document]):
        for node, doc in zip(random.choices(self.nodes, k=len(documents)), documents):
            node.add_doc(doc)
            node.update()

    def scatter_queries(self, queries: List[MessageQuery]):
        for node, query in zip(random.choices(self.nodes, k=len(queries)), queries):
            node.add_query(query)

    def run_embeddings(self, epochs):
        for time in range(epochs):
            random.shuffle(self.edges)
            for u, v in self.edges:
                if random.random() < 0.1:
                    mesg_to_v, mesg_to_u = u.send_embedding(), v.send_embedding()
                    v.receive_embedding(u, mesg_to_v)
                    u.receive_embedding(v, mesg_to_u)

    def run_queries(self, epochs, monitor):
        for time in range(epochs):
            random.shuffle(self.nodes)
            outgoing = {}
            for u in self.nodes:
                if u.has_queries_to_send(): # and random.random() < 0.8: # the probabilities does not matter
                    outgoing[u] = u.send_queries()
            for u, to_send in outgoing.items(): # TODO does it matter if synchronous?
                for v, queries in to_send.items():
                    v.receive_queries(queries, u)

            if monitor is not None and not monitor():
                break

    def __call__(self, epochs, monitor=None):
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
                if u.has_queries_to_send() and random.random() < 0.8:
                    outgoing[u] = u.send_queries()
            for u, to_send in outgoing.items(): # TODO does it matter if synchronous?
                for v, queries in to_send.items():
                    v.receive_queries(queries, u)

            if monitor is not None and not monitor():
                break
        return time
    


