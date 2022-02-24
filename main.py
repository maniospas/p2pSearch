from loader import *
import utils
from datatypes import Document, Query, MessageQuery
from simulation import DecentralizedSimulation
from nodes import *

import random
import networkx as nx
from matplotlib.colors import cnames


def test_print_network(edges, queries):
    graph = nx.Graph()
    graph.add_edges_from([(e[0].name, e[1].name) for e in edges])

    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="grey")
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edge_color="grey")

    colors = random.sample([cname for cname in cnames.keys() if cname.startswith("dark")], len(queries))
    for query, color in zip(queries, colors):
        nx.draw_networkx_edges(graph, pos, edgelist=query.visited_tree, edge_color=color, width=3)



ttl = 5

# load data
simulation = DecentralizedSimulation(load_graph(HardSumEmbeddingNode, "erdos"))
query_results = load_query_results()
que_embs = load_embeddings(dataset="glove", type="queries")
doc_embs = load_embeddings(dataset="glove", type="docs")
other_doc_embs = load_embeddings(dataset="glove", type="other_docs")

test_qids = random.sample(list(query_results.keys()), 4)
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
time = simulation(epochs=20, monitor=monitor)
print("Discovered everything at time", time)


test_print_network(simulation.edges, queries)