from nodes.base import DocNode
from collections import defaultdict
from datatypes import Document, MessageQuery

import random


class WalkerNode(DocNode):

    def __init__(self, name):
        super().__init__(name)
        self.seen_from = defaultdict(lambda : set())

    def receive_queries(self, queries, from_node):
        for query in queries:
            if query.name not in self.seen_from: # avoid rechecking if query (or clone) has been seen TODO rethink
                query.check_now(self.docs)

            if query.is_alive():
                if query.name in self.query_queue:
                    self.query_queue[query.name].receive(query)
                    query.kill()
                else:
                    self.query_queue[query.name] = query
            else:
                query.kill()
            self.seen_from[query.name].add(from_node)

    def send_queries(self):
        assert all([query.is_alive() for query in self.query_queue.values()]), "queries in query queue should have been alive"
        to_send = defaultdict(lambda : [])
        for query in self.query_queue.values():
            next_hops = self.get_next_hops(query)
            if len(next_hops) > 0:
                query.send()
                clones = [query.clone() for _ in range(len(next_hops)-1)]
                outgoing_queries = [query]
                outgoing_queries.extend(clones)
                for next_hop, outgoing_query in zip(next_hops, outgoing_queries):
                    to_send[next_hop].append(outgoing_query)
            else:
                query.kill()
        self.active_queries.clear()
        return to_send

    def get_next_hops(self, query):
        if len(self.neighbors) == 0:
            return [] # pathologic case

        candidates = set(self.neighbors).difference(self.seen_from[query.name])
        if len(candidates) > 0:
            return random.sample(candidates, k=1)
        else:
            return random.sample(list(self.neighbors), k=1)




