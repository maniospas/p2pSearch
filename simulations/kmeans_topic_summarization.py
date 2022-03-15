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

def draw_network_embs(graph):
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="yellow")
    nx.draw_networkx_labels(graph, pos, {node: str(node.personalization) for node in graph.nodes})
    nx.draw_networkx_edges(graph, pos, edge_color="grey")


random.seed(10)
ttl = 20
n_docs = 1000
n_iters = 1000
n_nodes_per_query = 5
dataset = "glove"
graph_name = "toy_erdos"
n_clusters = 30

# load data

dim, query_results, que_embs, doc_embs, other_doc_embs = load_all(dataset)
if n_clusters == 1:
    node_init = lambda name: HardSumEmbeddingNode(name, dim)
elif n_clusters > 1:
    cluster_centers = load_clusters(dataset, n_clusters)
    node_init = lambda name: ClusterSumNode(name, cluster_centers, dim)
else:
    raise(f"wtf {n_clusters}")
graph = load_graph(node_init, graph_name)
simulation = DecentralizedSimulation(graph)

print("create and scatter docs")
qid = random.choice(list(query_results.keys()))
gold_doc = Document(query_results[qid], doc_embs[query_results[qid]])
docs = [Document(doc_id, other_doc_embs[doc_id]) for doc_id in random.sample(list(other_doc_embs), n_docs-1)]
docs.append(gold_doc)
simulation.scatter_docs(docs)


# draw_network_embs(graph)

print("diffuse embeddings")
simulation.quick_embeddings()


queries = [Query(qid, que_embs[qid]) for _ in range(n_iters)]

for query in queries:
    nodes = simulation.sample_nodes(k=n_nodes_per_query)
    for node in nodes:
        node.add_query(MessageQuery(query, ttl))


def accuracy_monitor():
    acc = sum(1. for query in queries if query.candidate_doc == query_results[query.name]) / len(queries)

    print("Accuracy", acc)
    return acc < 0.99


def queue_monitor():
    n_messages = sum([len(node.query_queue) for node in simulation.nodes])
    print(f"{n_messages} message{'' if n_messages==1 else 's'} circulating")
    return n_messages > 0


print("begin simulation")
time = simulation.run_queries(50, monitor=accuracy_monitor)
print("Finished simulation at time", time)


# node_dict = {node.name: node for node in simulation.nodes}
#
# target_docids = [query_results[query.name] for query in queries]
# target_nodes = [[node for node in simulation.nodes if docid in node.docs][0] for docid in target_docids]
#
# test_print_network(simulation.edges, queries, target_nodes)