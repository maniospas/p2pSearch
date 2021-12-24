import numpy as np
import random as rnd
import numpy.random as nprnd

from datatypes import Text
from collections import defaultdict


def generate_random_unitary_embeddings(dim, n, norm_ord=2):
    embeddings = np.random.randn(n, dim)
    return embeddings / np.linalg.norm(embeddings, ord=norm_ord, axis=1, keepdims=True)


def get_random_texts(n, emb_dim, name=""):
    embs = generate_random_unitary_embeddings(dim=emb_dim, n=n)
    return [Text(f"{name} {i}", emb) for i, emb in enumerate(embs)]


def get_random_text_clones(n, emb_dim, name=""):
    emb = generate_random_unitary_embeddings(dim=emb_dim, n=1)[0]
    return [Text(f"{name} clone {i}", emb) for i in range(n)]


def get_random_assignment(nodes, resources):
    node2resources = defaultdict(lambda: [])
    for node, resource in zip(rnd.choices(nodes, k=len(resources)), resources):
        node2resources[node].append(resource)
    return node2resources


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


def set_seed(seed):
    rnd.seed(seed)
    nprnd.seed(seed)


def ttl_monitor(epoch, results):
    logs = [f"{query.name} has found {len(query.candidate_docs)} docs (ttl:{query.ttl})" for query in results]
    print(f"EPOCH {epoch}")
    print("\n".join(logs))
    print()
    return not all([query.ttl == 0 for query in results])
