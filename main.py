from loader import *
import utils
from datatypes import Document, ExchangedQuery
from simulation import DecentralizedSimulation
from nodes.base import BaseNode
import random


# load data
simulation = DecentralizedSimulation(load_graph(BaseNode))
query_results = load_query_results()
all_queries = load_embeddings(dataset="sts_benchmark", type="queries")
all_docs = load_embeddings(dataset="sts_benchmark", type="docs")
all_other_docs = load_embeddings(dataset="sts_benchmark", type="other_docs")

test_queries = random.sample(list(query_results.keys()), 100)

# assign query result docs to nodes
simulation.scatter_docs([Document(query_results[query], all_docs[query_results[query]]) for query in test_queries])

# sanity check that search is possible
assert sum(1 for query in test_queries if utils.search(simulation.nodes, all_queries[query]).name == query_results[query]) == len(test_queries)

# print("Warming up")
# simulation(epochs=20)

results = simulation.scatter_queries([ExchangedQuery(query, all_queries[query]) for query in test_queries])

def monitor():
    acc = sum(1. for query in results if query.candidate_doc == query_results[query.name]) / len(results)
    print("Accuracy", acc)
    return acc < 0.99

time = simulation(epochs=1000, monitor=monitor)
print("Discovered everything at time", time)
