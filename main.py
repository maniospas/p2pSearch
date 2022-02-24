from loader import *
import utils
from datatypes import Document, Query, MessageQuery
from simulation import DecentralizedSimulation
from nodes.walkers import WalkerNode
from nodes.flooders import FlooderNode
import random


ttl = 2

# load data
simulation = DecentralizedSimulation(load_graph(WalkerNode))
query_results = load_query_results()
que_embs = load_embeddings(dataset="glove", type="queries")
doc_embs = load_embeddings(dataset="glove", type="docs")
other_doc_embs = load_embeddings(dataset="glove", type="other_docs")

test_qids = random.sample(list(query_results.keys()), 100)
queries = [Query(qid, que_embs[qid]) for qid in test_qids]
docs = [Document(query_results[qid], doc_embs[query_results[qid]]) for qid in test_qids]

# assign query result docs to nodes
simulation.scatter_docs(docs)

# sanity check that search is possible
assert sum(1 for qid in test_qids if utils.search(simulation.nodes, que_embs[qid]).name == query_results[qid]) == len(test_qids)

print("Warming up")
simulation(epochs=20)

simulation.scatter_queries([MessageQuery(query, ttl) for query in queries])

def monitor():
    acc = sum(1. for query in queries if query.candidate_doc == query_results[query.name]) / len(queries)

    print("Accuracy", acc)
    return acc < 0.99

print("begin simulation")
time = simulation(epochs=100, monitor=monitor)
print("Discovered everything at time", time)
