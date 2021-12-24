import matplotlib.pyplot as plt
import numpy as np

from utils import *
from loader import *
from simulations.advertisement_sims import BaseSimulation
from nodes.walkers import *
from collections import defaultdict
from datatypes import Ranking


def compare(sim_ranking, gold_ranking):

    gold_docs_found = sum([1 for doc, _ in sim_ranking if doc in gold_ranking])
    return gold_docs_found / len(gold_ranking)


set_seed(1)
graph_name = "fb"
graph = load_graph(Node, graph_name)
nodes = list(graph.nodes)
dim = 768
# n_docs = 100
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

# docs = get_random_texts(n_docs, dim, "doc")
# queries = get_random_text_clones(n_sims, dim, "query")
capacity = 10


# gold ranking
doc_scores = [np.sum(doc.embedding * queries[0].embedding) for doc in docs]
gold_ranking = Ranking(docs, doc_scores, capacity)



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
init_nodes = [RandomWalkerWithHistory, MaxSimForwarder, MaxSimForwarderWithHistory]
labels = ["random walk",
          "biased walk without memory",
          "biased walk with memory"]

results = dict()
for init_node, label in zip(init_nodes, labels):
    for ttl in ttls:
        sim = BaseSimulation(**fixed_args, ttl=ttl, init_node=init_node)
        _, out = sim()
        results[(label, ttl)] = [compare(query.candidate_docs, gold_ranking) for query in out]

# plots
figsize=(5, 5)
label_font_properties = {"fontfamily": "sans-serif", "fontsize": 12}
title_font_properties = {"fontfamily": "sans-serif", "fontsize": 15}
colors = ["blue", "red", "darkorange"]
styles = ["-k", "-k", "-k"]
f, ax = plt.subplots(figsize=figsize)

ax.grid()

# ax.plot(ttls, [1 for _ in ttls], c="g", label="gold")

for color, label in zip(colors, labels):
    res = [results[(label, ttl)] for ttl in ttls]
    found_docs = np.mean(res, axis=1)

    # found_docs = np.quantile(res, 0.5, axis=1)
    # found_docs_upper_quantile = np.quantile(res, 0.75, axis=1)
    # found_docs_lower_quantile = np.quantile(res, 0.25, axis=1)

    ax.plot(ttls, found_docs, c=color, label=label)
    # ax.plot(ttls, found_docs_upper_quantile, "--", c=color, lw=1)
    # ax.plot(ttls, found_docs_lower_quantile, "--", c=color, lw=1)


ax.set_xlabel("Time-to-live (TTL)", label_font_properties)
ax.set_ylabel("Mean precision", label_font_properties)
ax.set_ylim(-0.05, 0.55)
ax.legend()
f.show()

# plt.savefig(f"mean_precision_{graph_name}.png")
