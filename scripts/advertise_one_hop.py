from loader import *
import utils
from datatypes import Text
from simulations import *
from nodes.walkers import Node

import random as rnd
import numpy.random as nprnd

import matplotlib.pyplot as plt
from collections import defaultdict


def get_random_texts(n, emb_dim, name=""):
    embs = utils.generate_random_unitary_embeddings(dim=emb_dim, n=n)
    return [Text(f"{name} {i}", emb) for i, emb in enumerate(embs)]


def monitor(epoch, results):
    logs = [f"{query.name} has found {len(query.candidate_docs)} docs (ttl:{query.ttl})" for query in results]
    print(f"EPOCH {epoch}")
    print("\n".join(logs))
    print()
    return not all([query.ttl == 0 for query in results])


seed = 1
dim = 768
n_docs = 1000
n_queries = 2
advertise_radius = 1
ttl = 20
capacity = 10
epochs = 1000

rnd.seed(seed)
nprnd.seed(seed)

args = {
    "graph": load_graph(Node),
    "docs": get_random_texts(n_docs, dim, "doc"),
    "query": get_random_texts(1, dim, "query")[0],
    "n_query_reps": n_queries,
    "advertise_radius": advertise_radius,
    "ttl": ttl,
    "capacity": capacity,
    "epochs": epochs,
    "monitor": monitor,
    "seed": seed,
    # "origin": None
}

ttls = range(2, 21, 2)
sims = [[SingleQuerySingleOriginSimulator(**args, ttl=ttl, advertise_radius=1, origin=node) for node in graph.nodes]
        for ttl in ttls]

outs = []
for sim in sims:
    _, results = sim()
    out = [len(q.candidate_docs) for q in results]
    outs.append(out)

f, ax = plt.subplots()
ax.boxplot(outs)
ax.set_xticklabels(ttls)
# ax.set_title("No advertisement")
# ax.set_ylabel("single origin")


# sims[0, 1] = [SingleQuerySingleOriginSimulator(**args, ttl=ttl, advertise_radius=1) for ttl in ttls]
# sims[1, 0] = [SingleQueryMultipleOriginSimulator(**args, ttl=ttl, advertise_radius=0) for ttl in ttls]
# sims[1, 1] = [SingleQueryMultipleOriginSimulator(**args, ttl=ttl, advertise_radius=1) for ttl in ttls]


# f, axes = plt.subplots(2,2)
# ax = axes[0, 0]
# ax.boxplot(all_results[2])
# ax.set_xticklabels(ttls)
# ax.set_title("No advertisement")
# ax.set_ylabel("single origin")
#
# ax = axes[0, 1]
# ax.boxplot(all_results[0])
# ax.set_xticklabels(ttls)
# ax.set_title("1 hop advertisement")
#
# ax = axes[1, 0]
# ax.boxplot(all_results[3])
# ax.set_xticklabels(ttls)
# ax.set_ylabel("random origins")
#
# ax = axes[1, 1]
# ax.boxplot(all_results[1])
# ax.set_xticklabels(ttls)



#
#
# data = []
# for ttl in ttls:
#     epoch, results = simulation(epochs=1000, nodes2queries=random_node2queries, ttl=ttl, capacity=capacity,
#                                 monitor=monitor)
#     print("Completed at epoch", epoch)
#     data.append([len(q.candidate_docs) for q in results])
# all_results.append(data)
#
# simulation.change_advertise_radius(0)
#
# data = []
# for ttl in ttls:
#     epoch, results = simulation(epochs=1000, nodes2queries=single_node2queries, ttl=ttl, capacity=capacity,
#                                 monitor=monitor)
#     print("Completed at epoch", epoch)
#     data.append([len(q.candidate_docs) for q in results])
# all_results.append(data)
#
# data = []
# for ttl in ttls:
#     epoch, results = simulation(epochs=1000, nodes2queries=random_node2queries, ttl=ttl, capacity=capacity,
#                                 monitor=monitor)
#     print("Completed at epoch", epoch)
#     data.append([len(q.candidate_docs) for q in results])
# all_results.append(data)
#


# doc_scores = np.sum(doc_embs * query_embs, axis=1)
# gold_ranking = Ranking(docs, doc_scores, capacity)
# sim_rankings = [query.candidate_docs for query in queries]
