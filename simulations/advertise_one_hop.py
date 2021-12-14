from loader import *
import utils
from datatypes import Document, TTLQuery
from simulation import DecentralizedSimulation
from nodes.randomwalk import RWNode
import random


# load data
nodeType = RWNode
simulation = DecentralizedSimulation(load_graph(nodeType))
query_results = load_query_results()

dim = 768
n_docs = 2000
n_queries = 3
ttl = 6
capacity = 10

doc_embs = utils.generate_random_unitary_embeddings(dim=dim, n=n_docs)
docs = [Document(f"doc {i}", emb) for i, emb in enumerate(doc_embs)]
query_embs = utils.generate_random_unitary_embeddings(dim=dim, n=n_queries)
queries = [TTLQuery(f"query {i}", emb, ttl, capacity) for i, emb in enumerate(query_embs)]

# assign query result docs to nodes
simulation.scatter_docs(docs)

print("Warming up")
simulation(epochs=20)
results = simulation.scatter_queries(queries)

def ttl_monitor():
    print("TTLs: " + ",".join( [str(query.ttl) for query in results] ))
    return not all([query.ttl == 0 for query in results])

time = simulation(epochs=1000, monitor=ttl_monitor)
print("Discovered everything at time", time)
