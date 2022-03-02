from nodes.base import Node
import random


class WalkerNode(Node):

    def receive_queries(self, queries, from_node):
        super().receive_queries(queries, from_node, kill_seen=False)

    def get_next_hops(self, query):
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        candidates = self.filter_seen_from(neighbors, query, as_type=list)
        if len(candidates) > 0:
            return random.sample(candidates, k=1)
        else:
            return random.sample(neighbors, k=1)


