import matplotlib.pyplot as plt
import numpy as np

from utils import *
from loader import *
from simulations.advertisement_sims import BaseSimulation
from nodes.walkers import *
from collections import defaultdict



set_seed(1)
graph_name = "erdos"
graph = load_graph(Node, graph_name)
nodes = list(graph.nodes)
dim = 768
n_sims = 200
dset_name = "antique_test_top10"
query2docs = load_query_results(dset_name)
doc_embs = load_embeddings(dset_name, "docs")
que_embs = load_embeddings(dset_name, "queries")

rand_query_names = rnd.sample(list(query2docs.keys()), 10)

queries = [Text(f"{rand_query_names[0]} clone {i}", que_embs[rand_query_names[2]]) for i in range(n_sims)]

doc_names = []
for query_name in rand_query_names:
    doc_names.extend(query2docs[query_name])
docs = [Text(f"{doc_name}", doc_embs[doc_name]) for doc_name in doc_names]
capacity = 10

fixed_args = {
    "graph": graph,
    "node2docs": get_random_assignment(nodes, docs),
    "node2queries": get_random_assignment(nodes, queries),
    "advertise_radius": 1,
    "capacity": 10,
    "epochs": 1000,
    "monitor": ttl_monitor,
}

ttls = range(2, 21, 2)
init_nodes = [RandomWalker, RandomWalkerWithHistory, MaxSimForwarder, MaxSimForwarderWithHistory]

labels = ["random walk without history",
          "random walk with history",
          "biased walk without history",
          "biased walk with history"]

results = dict()
for init_node, label in zip(init_nodes, labels):
    for ttl in ttls:
        sim = BaseSimulation(**fixed_args, ttl=ttl, init_node=init_node)
        _, out = sim()
        results[(label, ttl)] = [len(query.candidate_docs) for query in out]


# plots

label_font_properties ={"fontfamily":"serif", "fontsize":15}
colors = ["blue", "darkblue", "red", "darkred"]
f, ax = plt.subplots()

ax.grid()
for color, label in zip(colors, labels):
    res = [results[(label, ttl)] for ttl in ttls]
    # found_docs = np.mean(res, axis=1)

    found_docs = np.quantile(res, 0.5, axis=1)
    found_docs_upper_quantile = np.quantile(res, 0.75, axis=1)
    found_docs_lower_quantile = np.quantile(res, 0.25, axis=1)

    ax.plot(ttls, np.mean(res, axis=1), c=color, label=label)
    # ax.plot(ttls, found_docs_upper_quantile, "--", c=color, lw=1)
    # ax.plot(ttls, found_docs_lower_quantile, "--", c=color, lw=1)

ax.set_xlabel("TTL", label_font_properties)
ax.set_ylabel("Number of found documents", label_font_properties)
ax.legend()
f.show()


