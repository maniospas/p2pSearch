from nodes.base import DocNode
from collections import defaultdict


class FlooderNode(DocNode):

    def __init__(self, name):
        super().__init__(name)

    def receive_queries(self, queries, from_node):
        super().receive_queries(queries, from_node, kill_seen=True)

    def get_next_hops(self, query):
        if len(self.neighbors) == 0:
            return [] # pathologic case
        next_hops = set(self.neighbors).difference(self.seen_from[query.name])
        return list(next_hops)