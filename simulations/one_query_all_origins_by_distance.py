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

def stream_hops(graph, start_node, max_hop=float('inf')):
    hop = 0
    visited_nodes, current_nodes, next_nodes = set(), set(), {start_node}
    while len(next_nodes) > 0 and hop < max_hop:
        hop += 1
        yield list(next_nodes)
        current_nodes = next_nodes
        visited_nodes.update(next_nodes)
        next_nodes = set.union(*[set(graph.neighbors(node)) for node in current_nodes]).difference(visited_nodes)


def outcome_per_ttl(queries, query_results):
    assert len(queries) > 0
    acc = np.zeros((len(queries), queries[0].messages[0].ttl))
    for i, query in enumerate(queries):
        if query.candidate_doc == query_results[query.name]:
            acc[i, query.hops_to_reach_candidate_doc:] = 1
    return acc


def sim(graph_name, dataset_name, ppr_a, n_iters, ttl, n_docs, Q):

    print(f"{os.getpid()}: preparing" )
    dim, query_results, que_embs, doc_embs, other_doc_embs = load_all(dataset_name)
    graph = load_graph(lambda name: HardSumEmbeddingNode(name, dim, True), graph_name)
    adj = nx.adjacency_matrix(graph)
    ppr_mat = loader.analytic_ppr(adj, ppr_a, True, graph_name)
    HardSumEmbeddingNode.set_ppr_a(ppr_a)
    simulation = DecentralizedSimulation(graph, _graph_name=graph_name)

    node2neighbors_per_distance = dict()
    for origin_node in graph.nodes:
        node2neighbors_per_distance[origin_node] = dict(enumerate(stream_hops(graph, origin_node)))

    print(f"{os.getgid()}: start")
    hop2outcomes = defaultdict(lambda: [])
    for i in range(n_iters):
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
        hop2query = dict()
        for hop, nodes in node2neighbors_per_distance[test_node].items():
            node = random.choice(nodes)
            query = Query(test_qid, que_embs[test_qid], query_results[test_qid])
            node.add_query(query.spawn(ttl))
            hop2query[hop] = query

        # print("begin simulation")
        _ = simulation.run_queries(epochs=5 * ttl, monitor=None)
        # print("Finished search at time", time+1)

        for hop, query in hop2query.items():
            hop2outcomes[hop].append(query.candidate_doc == query_results[query.name])

        print(f"{os.getpid()}: ppr_a {ppr_a} iter {i + 1}/{n_iters} ({int(100 * (i + 1) / n_iters)}% complete)")

    Q.put({hop: np.mean(outcomes) for hop, outcomes in hop2outcomes.items()})


def big_sim():
    random.seed(10)
    np.random.seed(10)

    graph_name = "fb"
    dataset_name = "glove"
    ttl = 50
    # ppr_a = 0.5
    n_docs = 100
    n_iters = 5000

    all_ppr_a = [0.1, 0.5, 0.9]
    Q = multiprocessing.Queue()
    jobs = []
    for ppr_a in all_ppr_a:
        process = multiprocessing.Process(
            target=sim,
            args=(graph_name, dataset_name, ppr_a, n_iters, ttl, n_docs, Q)
        )
        jobs.append(process)

    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    print(f"----- finished! ----")
    all_hop2accs = [Q.get() for _ in range(Q.qsize())]
    Q.close()

    dirpath = os.path.join(os.path.dirname(__file__), "output", f"{n_docs}docs_{n_iters}iters_{ttl}ttl")
    with open(dirpath, "w") as f:
        for a, hop2accs in zip(all_ppr_a, all_hop2accs):
            hops = sorted(list(hop2accs.keys()))
            accs = [hop2accs[hop] for hop in hops]
            f.write(f"{a}\n")
            f.write(" ".join([str(hop) for hop in hops])+"\n")
            f.write(" ".join([str(acc) for acc in accs])+"\n")


big_sim()

n_docs = 100
n_iters = 5000
ttl=50
dirpath = os.path.join(os.path.dirname(__file__), "output", f"{n_docs}docs_{n_iters}iters_{ttl}ttl")
with open(dirpath, "r") as f:
    alpha2hopaccs = {}
    line = f.readline().rstrip()
    while line != "":
        alpha = float(line)
        hops = [int(token) for token in f.readline().rstrip().split()]
        accs = [float(token) for token in f.readline().rstrip().split()]
        alpha2hopaccs[alpha] = (hops, accs)
        line = f.readline().rstrip()

fig, ax = plt.subplots(figsize=(6, 5))
for alpha, marker in zip([0.1, 0.5, 0.9], ["+", "*", "o"]):
    hops, accs = alpha2hopaccs[alpha]
    ax.plot(hops, accs, "k-"+marker,label=rf"$\alpha = {alpha}$", ms=7, lw=1.0)
ax.grid()
ax.set_xlabel("TTL (hops)", family="serif", size=16)
ax.set_ylabel("Accuracy (%)", family="serif", size=16)
ax.legend(prop={'family':"serif", 'size': 13})

# from img import get_path
# path = get_path(__file__, f"{'fb'}_{'glove'}_{n_docs}docs.pdf")
# plt.savefig(path)
#
#
