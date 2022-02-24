from nodes import WalkerNode
from utils import softmax

import numpy as np


class HardSumEmbeddingNode(WalkerNode):

    def update(self):

        personalization = np.sum(doc.embedding for doc in self.docs.values())
        self.set_personalization(personalization)
        super().update()

    def get_next_hops(self, query):
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors[neighbor] for neighbor in neighbors]
        scores = np.array([np.sum(query.embedding * neighbor_embedding) for neighbor_embedding in neighbor_embeddings])
        idx = np.argmax(scores)
        return [neighbors[idx]]


class SoftSumEmbeddingNode(WalkerNode):

    def update(self):
        personalization = np.sum(doc.embedding for doc in self.docs.values())
        self.set_personalization(personalization)
        super().update()

    def get_next_hops(self, query):
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors[neighbor] for neighbor in neighbors]
        scores = np.array([np.sum(query.embedding * neighbor_embedding) for neighbor_embedding in neighbor_embeddings])
        idx = np.random.choice(neighbors, p=softmax(scores))
        return [neighbors[idx]]
