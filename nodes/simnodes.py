from nodes import WalkerNode

import numpy as np


class HardSumEmbeddingNode(WalkerNode):

    def __init__(self, name):
        super().__init__(name)

    def get_personalization(self):
        return np.sum(doc.embedding for doc in self.docs.values())

    def get_next_hops(self, query):
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        # filtered_neighbors = self.filter_seen_from(neighbors, query, as_type=list)
        # if len(filtered_neighbors) > 0:
        #     neighbors = filtered_neighbors
        #
        # filtered_neighbors = self.filter_sent_to(neighbors, query, as_type=list)
        # if len(filtered_neighbors) > 0:
        #     neighbors = filtered_neighbors

        filtered_neighbors = self.filter_query_history(neighbors, query, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors[neighbor] for neighbor in neighbors]
        scores = np.array([np.sum(query.embedding * neighbor_embedding) for neighbor_embedding in neighbor_embeddings])
        idx = np.argmax(scores)
        return [neighbors[idx]]


class SoftSumEmbeddingNode(WalkerNode):

    def get_personalization(self):
        return np.sum(doc.embedding for doc in self.docs.values())

    def get_next_hops(self, query):
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, query)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, query)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors[neighbor] for neighbor in neighbors]
        scores = np.array([np.sum(query.embedding * neighbor_embedding) for neighbor_embedding in neighbor_embeddings])
        idx = np.random.choice(np.argsort(-scores)[:3])  # choose randomly from top scored documents

        return [neighbors[idx]]
