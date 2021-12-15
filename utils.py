import numpy as np


def generate_random_unitary_embeddings(dim, n, norm_ord=2):
    embeddings = np.random.randn(n, dim)
    return embeddings / np.linalg.norm(embeddings, ord=norm_ord, axis=1, keepdims=True)


def topk(seq, k):
    pos = range(len(seq))
    rank = sorted(zip(pos, seq), key=lambda tup: tup[1], reverse=True)[:k]
    pos, vals = zip(*rank)
    return pos, vals


def search(nodes, query_embedding):
    return max([doc for node in nodes for doc in node.docs.values()],
               key=lambda doc: np.sum(doc.embedding * query_embedding))


def softmax(logits):
    logits = np.array(logits)
    probs = np.exp(logits)
    return probs / np.sum(probs)