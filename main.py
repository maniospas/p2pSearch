from loader import *
import utils
from datatypes import Document, ExchangedQuery
from simulation import DecentralizedSimulation
from nodes.flooding import FloodNode
import random


# load data
simulation = DecentralizedSimulation(load_graph(FloodNode))
query_results = load_query_results()
docs = load_embeddings(type="docs")
queries = load_embeddings(type="queries")
test_queries = random.sample(list(query_results.keys()), 3)

# assign query result docs to nodes
simulation.scatter_docs([Document(query_results[query], docs[query_results[query]]) for query in test_queries])

# sanity check that search is possible
assert sum(1 for query in test_queries if utils.search(simulation.nodes, queries[query]).name == query_results[query]) == len(test_queries)

print("Warming up")
simulation(epochs=20)
results = simulation.scatter_queries([ExchangedQuery(query, queries[query]) for query in test_queries])

def monitor():
    acc = sum(1. for query in results if query.candidate_doc == query_results[query.name]) / len(results)
    print("Accuracy", acc)
    return acc < 0.99

time = simulation(epochs=1000, monitor=monitor)
print("Discovered everything at time", time)
