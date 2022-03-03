from loader import *
import multiprocessing
from datatypes import Query
from simulation import DecentralizedSimulation
from nodes import *

import random
import numpy as np
import matplotlib.pyplot as plt


def calc_accuracy_per_ttl(queries):
    assert len(queries) > 0
    acc = np.zeros((len(queries), queries[0].messages[0].ttl))
    for i, query in enumerate(queries):
        if query.candidate_doc == query_results[query.name]:
            acc[i, query.hops_to_reach_candidate_doc:] = 1
    return np.mean(acc, axis=0)


def same_query_different_origins_simulation(graph_name, qid, gold_doc, n_docs, que_embs, other_doc_embs, result_queue):
    # create simulation data
    simulation = DecentralizedSimulation(load_graph(HardSumEmbeddingNode, graph_name))

    queries = [Query(qid, que_embs[qid]) for _ in range(len(simulation.nodes))] # one query per graph node
    docs = [Document(docid, other_doc_embs[docid]) for docid in random.sample(list(other_doc_embs), n_docs-1)] # change docs every time
    docs.append(gold_doc)

    # assign query result docs to nodes
    simulation.scatter_docs(docs)

    # # sanity check that search is possible
    # assert utils.search(simulation.nodes, que_embs[qid]).name == query_results[qid]

    print("Warming up")
    simulation.run_embeddings(epochs=200)

    for node, query in zip(simulation.nodes, queries):
        node.add_query(MessageQuery(query, ttl))

    def queue_monitor():
        n_messages = sum([len(node.query_queue) for node in simulation.nodes])
        print(f"{n_messages} message{'' if n_messages==1 else 's'} circulating")
        return n_messages > 0

    print("begin simulation")
    time = simulation.run_queries(epochs=50, monitor=queue_monitor)
    print("Finished seearch at time", time)

    acc = calc_accuracy_per_ttl(queries)
    result_queue.put(acc)

random.seed(10)

graph_name = "fb"
dataset_name = "glove"
n_docs = 10
ttl = 20
n_iters = 10


# load ir dataset
query_results = load_query_results()
que_embs = load_embeddings(dataset=dataset_name, type="queries")
doc_embs = load_embeddings(dataset=dataset_name, type="docs")
other_doc_embs = load_embeddings(dataset=dataset_name, type="other_docs")

qid = random.choice(list(query_results)) # use the same query for all simulations
gold_doc = Document(query_results[qid], doc_embs[query_results[qid]])

Q = multiprocessing.Queue()
jobs = []
for i in range(n_iters):
    process = multiprocessing.Process(
        target=same_query_different_origins_simulation,
        args=(graph_name, qid, gold_doc, n_docs, que_embs, other_doc_embs, Q)
    )
    jobs.append(process)

for j in jobs:
    j.start()

for j in jobs:
    j.join()

all_accs = np.array([Q.get() for _ in range(Q.qsize())])
Q.close()

accs_median = np.median(all_accs, axis=0)
accs_q1 = np.percentile(all_accs, 75, axis=0)
accs_q3 = np.percentile(all_accs, 25, axis=0)

fig, ax = plt.subplots()
ax.plot(range(ttl), accs_median, "b-")
ax.plot(range(ttl), accs_q1, "b--")
ax.plot(range(ttl), accs_q3, "b--")
ax.grid()
ax.set_xlabel("TTL")
ax.set_ylabel("Accuracy")
plt.show