import utils
from datatypes import Document, TTLQuery
import numpy as np
from collections import defaultdict


class Node:

    def __init__(self, name):
        self.name = name
        self.docs = dict()

        self.neighbors = dict()
        self.active_queries = dict()
        self.seen_queries = dict()

    def _update_active_queries(self, query):
        if query.name in self.active_queries:
            self.active_queries[query.name].receive(query)
            query.kill()
        else:
            self.active_queries[query.name] = query

    def _update_seen_queries(self, query):
        if query.name in self.seen_queries:
            self.seen_queries[query.name].receive(query)
        else:
            self.seen_queries[query.name] = query

    def learn_neighbor(self, node, embedding=0):
        if embedding is None:
            embedding = 0.0
        self.neighbors[node] = embedding

    @property
    def embedding(self):
        embs = np.array([doc.embedding for doc in self.docs.values()])

        return np.sum(embs, axis=0)

    def add_doc(self, doc: Document):
        self.docs[doc.name] = doc

    def add_query(self, query):
        self.active_queries[query.name] = query

    def update(self):
        pass

    def send(self, neighbor):
        pass

    def receive(self, neighbor, queries):
        for query in queries:

            self._update_seen_queries(query)

            query.check_now(self.docs)

            if query.is_alive():
                self._update_active_queries(query)
            else:
                query.kill()
                print(f"query {query.name} died at node {self.name}")

    def forward(self):

        to_forward = defaultdict(lambda: [])
        neighbors, neighbor_embs = zip(*self.neighbors.items())
        for query in self.active_queries.values():
            neighbor_scores = [np.sum(neighbor_emb * query.embedding) for neighbor_emb in neighbor_embs]
            neighbor_forward_probs = utils.softmax(neighbor_scores)
            if len(neighbors) == 0:
                query.kill()
                continue
            next_hop = np.random.choice(neighbors, p=neighbor_forward_probs)
            query.send((self, next_hop))
            to_forward[next_hop].append(query)

        self.active_queries.clear()

        return to_forward

    def clear_docs(self):
        self.docs = dict()

    def clear_neighbors(self):
        self.neighbors = dict()

    def clear_queries(self):
        self.seen_queries = dict()
        self.active_queries = dict()

    def __str__(self):
        return f"{self.__class__.__name__.lower() } {self.name}"

    def __repr__(self):
        return str(self)


if __name__ == "__main__":
    amy, katia, guadelupe = Node("Amy"), Node("Katia"), Node("Guadelupe")
    docs = [
        Document("1", np.array([0.11,0.2,0.3])),
        Document("2", np.array([0.5,0.5,0.4])),
        Document("3", np.array([0.1,0.5,0.9])),
        ]
    amy.add_doc(docs[0])
    amy.add_doc(docs[1])
    # guadelupe.add_doc(docs[2])
    katia.add_doc(docs[0])

    amy.learn_neighbor(katia, katia.embedding)
    amy.learn_neighbor(guadelupe, guadelupe.embedding)
    katia.learn_neighbor(amy, amy.embedding)
    guadelupe.learn_neighbor(amy, amy.embedding)

    q = TTLQuery( "query", np.array([-1,-1,9]), 2, 10, candidate_docs=None)
    amy.add_query(q)

    res = amy.forward()