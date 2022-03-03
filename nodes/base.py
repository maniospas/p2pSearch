from abc import ABC, abstractmethod

import numpy as np

from datatypes import Document, MessageQuery
from collections import defaultdict


class Node:

    def __init__(self, name, ppr_a, init_personalization=lambda: 0):
        self.name = name
        self.ppr_a = ppr_a
        self.neighbors = dict()
        self.personalization = init_personalization()
        self.embedding = self.personalization

        self.docs = dict()
        self.query_queue = dict()
        self.seen_from = defaultdict(lambda: set())
        self.sent_to = defaultdict(lambda: set())

    def add_doc(self, doc: Document):
        self.docs[doc.name] = doc

        self.personalization = self.get_personalization()
        self.embedding = self.personalization

    def add_query(self, query: MessageQuery):
        assert query.ttl >= 0, f"{query}, ttl should be >= 0"
        query.check_now(self.docs)
        if query.is_alive():
            if query.name in self.query_queue:
                self.query_queue[query.name].receive(query)
                query.kill(self, reason=f"query merged with queued clone")
            else:
                self.query_queue[query.name] = query
        else:  # if ttl was initially 0
            query.kill(self, reason="ttl was initialized to 0")

    def filter_seen_from(self, nodes, query, as_type=list):
        return as_type(set(nodes).difference(self.seen_from[query.name]))

    def filter_sent_to(self, nodes, query, as_type=list):
        return as_type(set(nodes).difference(self.sent_to[query.name]))

    def filter_query_history(self, nodes, query, as_type=list):
        nodes = {node.name: node for node in nodes}
        for visited_node_name in query.visited_nodes:
            if visited_node_name in nodes:
                nodes.pop(visited_node_name)
        return as_type(nodes.values())

    def has_queries_to_send(self):
        return len(self.query_queue) > 0

    def send_embedding(self):
        return self.embedding / max(1, len(self.neighbors)) ** 0.5

    @DeprecationWarning
    def update_embedding(self):
        embedding = sum([emb for emb in self.neighbors.values()])
        self.embedding = self.ppr_a * self.personalization + (1-self.ppr_a) * embedding / len(self.neighbors)**0.5

    def receive_embedding(self, neighbor, neighbor_embedding):

        N = len(self.neighbors)
        if neighbor in self.neighbors:
            self.embedding += (neighbor_embedding - self.neighbors[neighbor]) / N**0.5 * (1-self.ppr_a)
        else:
            self.embedding = ((self.embedding - self.ppr_a * self.personalization) * N**0.5
                              + neighbor_embedding * (1-self.ppr_a)) / (N+1)**0.5 + self.ppr_a * self.personalization
        self.neighbors[neighbor] = neighbor_embedding
        # self.update_embedding()

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
                    self.sent_to[outgoing_query.name].add(next_hop)
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

    @abstractmethod
    def get_personalization(self):
        pass
