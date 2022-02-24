from nodes.base import DocNode
from collections import defaultdict
from datatypes import Document, MessageQuery

import random
import numpy as np


class WalkerNode(DocNode):

    def __init__(self, name):
        super().__init__(name)

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


