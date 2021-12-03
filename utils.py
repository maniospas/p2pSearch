import numpy as np


def search(nodes, query_embedding):
    return max([doc for node in nodes for doc in node.docs.values()], key=lambda doc: np.sum(doc.embedding * query_embedding))
