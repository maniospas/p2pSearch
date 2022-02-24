from nodes.base import DocNode
from collections import defaultdict
from datatypes import Document, MessageQuery

import random


class WalkerNode(DocNode):

    def __init__(self, name):
        super().__init__(name)

    def receive_queries(self, queries, from_node):
        super().receive_queries(queries, from_node, kill_seen=False)

    def get_next_hops(self, query):
        if len(self.neighbors) == 0:
            return [] # pathologic case

        candidates = set(self.neighbors).difference(self.seen_from[query.name])
        if len(candidates) > 0:
            return random.sample(candidates, k=1)
        else:
            return random.sample(list(self.neighbors), k=1)




