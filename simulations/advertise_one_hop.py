from loader import *
import utils
from datatypes import Document, TTLQuery, Ranking
from simulation import RandomWalkWithAdvertisementsSimulation
from nodes.randomwalk import Node
# import random

import numpy as np


def random_docs(n_docs, emb_dim):
    doc_embs = utils.generate_random_unitary_embeddings(dim=emb_dim, n=n_docs)
    return [Document(f"doc {i}", emb) for i, emb in enumerate(doc_embs)]


def random_ttl_queries(n_queries, emb_dim):
    query_embs = utils.generate_random_unitary_embeddings(dim=emb_dim, n=n_queries)
    return [TTLQuery(f"query {i}", emb, ttl, capacity) for i, emb in enumerate(query_embs)]


def monitor(epoch, results):
    logs = [f"{query.name} has found {len(query.candidate_docs)} docs (ttl:{query.ttl})" for query in results]
    print(f"EPOCH {epoch}")
    print("\n".join(logs))
    print()
    return not all([query.ttl == 0 for query in results])


dim = 768
n_docs = 50
n_queries = 20
ttl = 5
capacity = 10
advertise_radius = 5


graph = load_graph(Node)

doc_embs = utils.generate_random_unitary_embeddings(dim, n_docs)
docs = [Document(f"doc {i}", emb) for i, emb in enumerate(doc_embs)]

query_embs = utils.generate_random_unitary_embeddings(dim, n_queries)
queries = [Document(f"query {i}", query_emb) for i, query_emb in enumerate(query_embs)]

simulation = RandomWalkWithAdvertisementsSimulation(graph, docs, advertise_radius)


querying_node = simulation.sample_nodes(1)[0]
epoch, results = simulation(epochs=1000, nodes2queries={querying_node: queries}, ttl=ttl, capacity=capacity, monitor=monitor)
print("Completed at epoch", epoch)

# doc_scores = np.sum(doc_embs * query_embs, axis=1)
# gold_ranking = Ranking(docs, doc_scores, capacity)
# sim_rankings = [query.candidate_docs for query in queries]
