# import utils
import numpy as np
import random as rnd

from datatypes import Document
from collections import defaultdict


class Node:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__} {self.name}"

    def __repr__(self):
        return str(self)


class DocumentNode(Node):

    def __init__(self, name):
        super().__init__(name)

        self.docs = dict()
        self.neighbors_info = dict()
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

    def learn_neighbor(self, node, info=None):
        self.neighbors_info[node] = info

    @property
    def embedding(self):
        if len(self.docs) == 0:
            return None
        else:
            embs = np.array([doc.embedding for doc in self.docs.values()])
            return np.sum(embs, axis=0)

    def add_doc(self, doc: Document):
        self.docs[doc.name] = doc

    def add_query(self, query):
        self.active_queries[query.name] = query

    def receive(self, neighbor, queries):
        for query in queries:

            self._update_seen_queries(query)

            query.check_now(list(self.docs.values()))

            if query.is_alive():
                self._update_active_queries(query)
            else:
                query.kill()
                print(f"query {query.name} died at node {self.name}")

    def forward(self):

        to_forward = defaultdict(lambda: [])

        if len(self.neighbors_info) == 0:
            for query in self.active_queries.values():
                query.kill()  # TODO dependency for monitor, there shouldn't be a need to kill
        else:
            for query in self.active_queries.values():
                next_hop = self.get_next_hop(query)
                query.send((self, next_hop))
                to_forward[next_hop].append(query)

        self.active_queries.clear()
        return to_forward

    def get_next_hop(self, query):
        candidates = [node for node in self.neighbors_info if node not in query.visited_nodes]
        if len(candidates) == 0:
            candidates = list(self.neighbors_info.keys())
        return rnd.choice(candidates)

    def __str__(self):
        return f"Docnode {self.name}"

    def __repr__(self):
        return str(self)


class RandomWalker(DocumentNode):

    def get_next_hop(self, query):
        candidates = [node for node in self.neighbors_info
                      if node not in query.visited_nodes[-1:]]
        if len(candidates) == 0:
            candidates = list(self.neighbors_info.keys())
        return rnd.choice(candidates)


class RandomWalkerWithHistory(DocumentNode):

    def get_next_hop(self, query):
        candidates = [node for node in self.neighbors_info if node not in query.visited_nodes]
        if len(candidates) == 0:
            candidates = list(self.neighbors_info.keys())
        return rnd.choice(candidates)


class MaxSimForwarder(DocumentNode):

    def get_next_hop(self, query):
        candidates = {neighbor: emb for neighbor, emb in self.neighbors_info.items()
                      if neighbor not in query.visited_nodes[-1:]}
        if len(candidates) == 0:
            candidates = self.neighbors_info

        cand_nodes, cand_embs = zip(*candidates.items())
        cand_scores = [-1 if cand_emb is None else np.sum(cand_emb * query.embedding)
                       for cand_emb in cand_embs]
        return cand_nodes[np.argmax(cand_scores)]


class MaxSimForwarderWithHistory(DocumentNode):

    def get_next_hop(self, query):
        candidates = {neighbor: emb for neighbor, emb in self.neighbors_info.items()
                      if neighbor not in query.visited_nodes}
        if len(candidates) == 0:
            candidates = self.neighbors_info

        cand_nodes, cand_embs = zip(*candidates.items())
        cand_scores = [-1 if cand_emb is None else np.sum(cand_emb * query.embedding)
                       for cand_emb in cand_embs]
        return cand_nodes[np.argmax(cand_scores)]
