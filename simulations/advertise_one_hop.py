from loader import *
import utils
from datatypes import Document, TTLQuery, Ranking
from simulation import SynchronousDecentralizedSimulation
from nodes.randomwalk import Node
# import random

import numpy as np


def random_docs(n_docs, emb_dim):
    doc_embs = utils.generate_random_unitary_embeddings(dim=emb_dim, n=n_docs)
    return [Document(f"doc {i}", emb) for i, emb in enumerate(doc_embs)]


def random_ttl_queries(n_queries, emb_dim):
    query_embs = utils.generate_random_unitary_embeddings(dim=emb_dim, n=n_queries)
    return [TTLQuery(f"query {i}", emb, ttl, capacity) for i, emb in enumerate(query_embs)]


# load data
node_type = Node
simulation = SynchronousDecentralizedSimulation(load_graph(node_type))
query_results = load_query_results()

dim = 768
n_docs = 7000
n_queries = 1
ttl = 10
capacity = 10

doc_embs = utils.generate_random_unitary_embeddings(dim, n_docs)
docs = [Document(f"doc {i}", emb) for i, emb in enumerate(doc_embs)]
simulation.scatter_docs(docs)

query_emb = utils.generate_random_unitary_embeddings(dim, 1)[0]
queries = [TTLQuery(f"query clone {i}", query_emb, ttl, capacity) for i in range(n_queries)]
results = simulation.scatter_queries(queries)


def ttl_monitor():
    print("TTLs: " + ",".join( [str(query.ttl) for query in results] ))
    return not all([query.ttl == 0 for query in results])


time = simulation(epochs=1000, monitor=ttl_monitor)
print("Completed at time", time)


doc_scores = np.sum(doc_embs * query_emb, axis=1)
gold_ranking = Ranking(docs, doc_scores, capacity)
sim_rankings = [query.candidate_docs for query in results]
