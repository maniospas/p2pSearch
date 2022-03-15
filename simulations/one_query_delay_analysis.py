import os

from loader import *
import multiprocessing
from datatypes import Query
from simulation import DecentralizedSimulation
from nodes import *

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time


def outcome_per_ttl(queries, query_results):
    assert len(queries) > 0
    acc = np.zeros((len(queries), queries[0].messages[0].ttl))
    for i, query in enumerate(queries):
        if query.candidate_doc == query_results[query.name]:
            acc[i, query.hops_to_reach_candidate_doc:] = 1
    return acc


def sim(graph_name, dataset_name, ppr_a, n_iters, n_docs):

    print(f"preparing" )
    dim, query_results, que_embs, doc_embs, other_doc_embs = load_all(dataset_name)
    graph = load_graph(lambda name: HardSumEmbeddingNode(name, dim, True), graph_name)
    adj = nx.adjacency_matrix(graph)
    ppr_mat = loader.analytic_ppr(adj, ppr_a, True, graph_name)
    HardSumEmbeddingNode.set_ppr_a(ppr_a)
    simulation = DecentralizedSimulation(graph, _graph_name=graph_name)

    print(f"start")
    all_hops = []
    n_succeeded = 0
    for i in range(n_iters):
        print(f"iter {i+1}/{n_iters}")
        simulation.clear()  # TODO: dangerous

        # print("create docs")
        test_qid = random.choice(list(query_results))
        test_doc = Document(query_results[test_qid], doc_embs[query_results[test_qid]])
        other_docs = [Document(docid, other_doc_embs[docid]) for docid in
                      random.sample(list(other_doc_embs), n_docs - 1)]

        # print("scatter docs")
        test_node = simulation.sample_nodes(1)[0]
        test_node.add_doc(test_doc)
        simulation.scatter_docs(other_docs)

        # print("computing embeddings")
        simulation.quick_embeddings(ppr_mat)

        # print("scatter queries")
        queries = []
        for node in simulation.sample_nodes(10):
            query = Query(test_qid, que_embs[test_qid], query_results[test_qid])
            node.add_query(query.spawn(ttl))
            queries.append(query)

        # print("begin simulation")
        _ = simulation.run_queries(epochs=5 * ttl, monitor=None)
        # print("Finished search at time", time+1)

        succeeded = [query for query in queries if query.candidate_doc == query_results[query.name]]
        hops = [query.hops_to_reach_candidate_doc for query in succeeded]

        all_hops.extend(hops)
        n_succeeded += len(succeeded)

    return n_succeeded, n_succeeded / (n_iters * .nodes)), all_hops



    # dirpath = os.path.join(os.path.dirname(__file__), "output", f"{n_docs}docs_{n_iters}iters_{ttl}ttl")
    # with open(dirpath, "w") as f:
    #     for a, hop2accs in zip(all_ppr_a, all_hop2accs):
    #         hops = sorted(list(hop2accs.keys()))
    #         accs = [hop2accs[hop] for hop in hops]
    #         f.write(f"{a}\n")
    #         f.write(" ".join([str(hop) for hop in hops])+"\n")
    #         f.write(" ".join([str(acc) for acc in accs])+"\n")


n_docs = 100
n_iters = 500
ttl = 50
graph_name = "fb"
dataset_name = "glove"
ppr_a = 0.5
n_docs = 10000

n_success, p_success, hops = sim(graph_name, dataset_name, ppr_a, n_iters, n_docs)
print("finished!")

print(n_success)

# dirpath = os.path.join(os.path.dirname(__file__), "output", f"{n_docs}docs_{n_iters}iters_{ttl}ttl")
# with open(dirpath, "r") as f:
#     alpha2hopaccs = {}
#     line = f.readline().rstrip()
#     while line != "":
#         alpha = float(line)
#         hops = [int(token) for token in f.readline().rstrip().split()]
#         accs = [float(token) for token in f.readline().rstrip().split()]
#         alpha2hopaccs[alpha] = (hops, accs)
#         line = f.readline().rstrip()
#
# fig, ax = plt.subplots(figsize=(6, 5))
# for alpha, marker in zip([0.1, 0.5, 0.9], ["+", "*", "o"]):
#     hops, accs = alpha2hopaccs[alpha]
#     ax.plot(hops, accs, "k-"+marker,label=rf"$\alpha = {alpha}$", ms=7, lw=1.0)
# ax.grid()
# ax.set_xlabel("TTL (hops)", family="serif", size=16)
# ax.set_ylabel("Accuracy (%)", family="serif", size=16)
# ax.legend(prop={'family':"serif", 'size': 13})
#
# from img import get_path
# path = get_path(__file__, f"{'fb'}_{'glove'}_{n_docs}docs.pdf")
# plt.savefig(path)


