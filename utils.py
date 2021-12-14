import numpy as np


def generate_random_unitary_embeddings(dim, n):
    embeddings = np.random.randn(n, dim)
    return embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)


def search(nodes, query_embedding):
    return max([doc for node in nodes for doc in node.docs.values()],
               key=lambda doc: np.sum(doc.embedding * query_embedding))
