from .advertising import AdvertisingNode
from datatypes import Document, ExchangedQuery
import numpy as np


class RWNode(AdvertisingNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docs = dict() # filled by external code
        self.queries = dict() # filled by external code (may be empty at the beginning)
        # self.queries_next_hop = dict()

    def add_doc(self, doc: Document):
        self.docs[doc.name] = doc

    def add_query(self, query: ExchangedQuery):
        self.queries[query.name] = query

    def update(self):
        personalization = 0
        for doc in self.docs.values():
            personalization = personalization + doc.embedding #/ len(self.docs)
        self.set_personalization(personalization)
        super().update()

    def send(self, neighbor):
        self.neighbors.setdefault(neighbor, 0) # put zero embedding if node is not known
        queries_to_send = []
        for query in self.queries.values():
            neighbor_score = np.sum(self.neighbors[neighbor] * query.embedding)
            all_neighbor_scores = [np.sum(neigh_emb * query.embedding) for neigh_emb in self.neighbors.values()]
            neighbor_prob = np.exp(neighbor_score) / np.sum(np.exp(all_neighbor_scores))
            if np.random.rand() < neighbor_prob:
                query = query.send()
                if query is not None:
                    queries_to_send.append(query)

        return super().send(neighbor), queries_to_send

    def receive(self, neighbor, message):
        neighbor_embedding, neighbor_queries = message
        for query in neighbor_queries:
            if query.name in self.queries:
                self.queries[query.name].receive(query)
            else:
                self.queries[query.name] = query
        for query in self.queries.values():
            query.check_now(self.docs)
        super().receive(neighbor, neighbor_embedding)
