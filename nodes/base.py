from abc import ABC, abstractmethod
from datatypes import Document, MessageQuery
from collections import defaultdict


class PPRNode:
    def __init__(self, name):
        self.name = name
        self.neighbors = dict()
        self.personalization = 0
        self.embedding = self.personalization

    def set_personalization(self, personalization):
        self.personalization = personalization
        self.embedding = personalization # also resets embedding

    def update(self):
        if len(self.neighbors) == 0:
            return
        embedding = 0
        for neighbor_embedding in self.neighbors.values():
            embedding = embedding + neighbor_embedding
        self.embedding = embedding / len(self.neighbors)**0.5 * 0.9 + self.personalization * 0.1

    def receive_embedding(self, neighbor, neighbor_embedding):
        self.neighbors[neighbor] = neighbor_embedding
        self.update()

    def send_embedding(self):
        return self.embedding / max(1, len(self.neighbors)) ** 0.5

    def __repr__(self):
        return f"Node ({self.name})"


class DocNode(PPRNode, ABC):

    def __init__(self, name):
        super(DocNode, self).__init__(name)
        self.docs = dict()
        self.query_queue = dict()
        self.seen_from = defaultdict(lambda : set())

    def add_doc(self, doc: Document):
        self.docs[doc.name] = doc

    def add_query(self, query: MessageQuery):
        assert query.ttl >= 0, f"{query}, ttl should be >= 0"
        query.check_now(self.docs)
        if query.is_alive():
            if query.name in self.query_queue:
                self.query_queue[query.name].receive(query)
                query.kill(self, reason=f"query merged with queued clone")
            else:
                self.query_queue[query.name] = query
        else: # if ttl was initially 0
            query.kill(self, reason="ttl was initialized to 0")

    def has_queries_to_send(self):
        return len(self.query_queue) > 0

    def send_queries(self):
        assert all([query.is_alive() for query in self.query_queue.values()]), "queries in query queue should have been alive"
        to_send = defaultdict(lambda : [])
        for query in self.query_queue.values():
            next_hops = self.get_next_hops(query)
            if len(next_hops) > 0:
                clones = [query.clone() for _ in range(len(next_hops)-1)]
                outgoing_queries = [query]
                outgoing_queries.extend(clones)
                for next_hop, outgoing_query in zip(next_hops, outgoing_queries):
                    outgoing_query.send(self, next_hop)
                    to_send[next_hop].append(outgoing_query)
            else:
                query.kill(self, reason="no next hops to forward")
        self.query_queue.clear()
        return to_send

    def receive_queries(self, queries, from_node, kill_seen=False):
        for query in queries:
            if query.name not in self.seen_from:
                query.check_now(self.docs)
            elif query.name in self.seen_from and kill_seen:
                query.kill(self, reason="query has already been seen")

            if query.is_alive():
                if query.name in self.query_queue:
                    self.query_queue[query.name].receive(query)
                    query.kill(self, reason=f"query merged at node {self.name}")
                else:
                    self.query_queue[query.name] = query
            else:
                query.kill(self, reason="query reached its ttl limit")
            self.seen_from[query.name].add(from_node)

    @abstractmethod
    def get_next_hops(self, query):
        pass
