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

    # TODO can remove nodes where we have already sent the message or see message history
    def get_next_hops(self, query):
        if len(self.neighbors) == 0:
            return [] # pathologic case

        candidates = self.filter_seen_from(self.neighbors)
        if len(candidates) > 0:
            return random.sample(candidates, k=1)
        else:
            return random.sample(list(self.neighbors), k=1)


