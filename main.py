from loader import *
import utils
from datatypes import Document, Query, MessageQuery
from simulation import DecentralizedSimulation
from nodes import *

import random
import networkx as nx
from matplotlib.colors import cnames


def test_print_network(edges, queries, target_nodes):
    graph = nx.Graph()
    graph.add_edges_from([(e[0].name, e[1].name) for e in edges])

    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="grey")
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edge_color="grey")

    colors = random.sample([cname for cname in cnames.keys() if cname.startswith("dark")], len(queries))
    for query, target_node, color in zip(queries, target_nodes, colors):
        nx.draw_networkx_edges(graph, pos, edgelist=query.visited_tree, edge_color=color, width=3)
        nx.draw_networkx_nodes(graph, pos, nodelist=[target_node.name], node_color=color)


random.seed(10)
ttl = 5

# load data
simulation = DecentralizedSimulation(load_graph(HardSumEmbeddingNode, "erdos"))
query_results = load_query_results()
que_embs = load_embeddings(dataset="glove", type="queries")
doc_embs = load_embeddings(dataset="glove", type="docs")
other_doc_embs = load_embeddings(dataset="glove", type="other_docs")

test_qids = random.sample(list(query_results.keys()), 1)
queries = [Query(qid, que_embs[qid]) for qid in test_qids]
docs = [Document(query_results[qid], doc_embs[query_results[qid]]) for qid in test_qids]
docs.extend([Document(doc_id, other_doc_embs[doc_id]) for doc_id in random.sample(list(other_doc_embs), 100)])

# assign query result docs to nodes
simulation.scatter_docs(docs)

# sanity check that search is possible
assert sum(1 for qid in test_qids if utils.search(simulation.nodes, que_embs[qid]).name == query_results[qid]) == len(test_qids)

print("Warming up")


nodes = simulation.sample_nodes(k=len(queries))

simulation(epochs=200)

for node, query in zip(nodes, queries):
    node.add_query(MessageQuery(query,ttl))

    # simulation.scatter_queries([MessageQuery(query, ttl) for query in queries])

def accuracy_monitor():
    acc = sum(1. for query in queries if query.candidate_doc == query_results[query.name]) / len(queries)

    print("Accuracy", acc)
    return acc < 0.99

def queue_monitor():
    n_messages = sum([len(node.query_queue) for node in simulation.nodes])
    print(f"{n_messages} message{'' if n_messages==1 else 's'} circulating")
    return n_messages > 0

print("begin simulation")
time = simulation(epochs=20, monitor=accuracy_monitor)
print("Discovered everything at time", time)

node_dict = {node.name: node for node in simulation.nodes}

target_docids = [query_results[query.name] for query in queries]
target_nodes = [[node for node in simulation.nodes if docid in node.docs][0] for docid in target_docids]

test_print_network(simulation.edges, queries, target_nodes)