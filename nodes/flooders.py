from nodes.base import Node


class FlooderNode(Node):

    def __init__(self, name):
        super().__init__(name)

    def receive_queries(self, queries, from_node):
        super().receive_queries(queries, from_node, kill_seen=True)

    def get_next_hops(self, query):
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        next_hops = self.filter_seen_from(neighbors, query, as_type=list)
        return next_hops

    def get_personalization(self):
        return None  # of no importance to flooder
