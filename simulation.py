import random
from datatypes import Document, MessageQuery
from typing import List


class DecentralizedSimulation:
    def __init__(self, graph):
        self.nodes = list(graph.nodes)
        self.edges = list(graph.edges())

    def scatter_docs(self, documents: List[Document]):
        for node, doc in zip(random.sample(self.nodes, len(documents)), documents):
            node.add_doc(doc)
            node.update()

    def scatter_queries(self, queries: List[MessageQuery]):
        for node, query in zip(random.sample(self.nodes, len(queries)), queries):
            node.add_query(query)


    def warmup(self, epochs, monitor=None):
        for time in range(epochs):
            random.shuffle(self.edges)
            for u, v in self.edges:
                if random.random() < 0.1:
                    mesg_to_v, mesg_to_u = u.send_embedding(v), v.send_embedding(u)
                    v.receive_embedding(u, mesg_to_v)
                    u.receive_embedding(v, mesg_to_u)

            for u in self.node:
                if u.has_queries_to_send():
                    if random.random() < 0.8:
                        to_send = u.send_queries()


            if monitor is not None and not monitor():
                break
        return time
    


