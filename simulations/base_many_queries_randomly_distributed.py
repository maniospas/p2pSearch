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
ttl = 20
n_docs = 10
n_iters = 1000
n_nodes_per_query = 5

# load data

query_results = load_query_results()
que_embs = load_embeddings(dataset="glove", type="queries")
doc_embs = load_embeddings(dataset="glove", type="docs")
other_doc_embs = load_embeddings(dataset="glove", type="other_docs")
dim = len(next(iter(other_doc_embs.values())))
node_init = lambda name: HardSumEmbeddingNode(name, init_personalization=lambda: np.zeros(dim))
simulation = DecentralizedSimulation(load_graph(node_init, "fb"))

print("create and scatter docs")
qid = random.choice(list(query_results.keys()))
gold_doc = Document(query_results[qid], doc_embs[query_results[qid]])
docs = [Document(doc_id, other_doc_embs[doc_id]) for doc_id in random.sample(list(other_doc_embs), n_docs-1)]
docs.append(gold_doc)
simulation.scatter_docs(docs)

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