import numpy as np
import random as rand
import matplotlib.pyplot as plt
from matplotlib.colors import cnames


from nodes import HardSumEmbeddingNode, HardSumL2EmbeddingNode, HardSumL2EmbeddingNodeWithSpawn
from loader import load_graph, load_node2vec
from simulation import DecentralizedSimulation
from utils import random_unitary, unitary
from datatypes import Document, Query, MessageQuery
import networkx as nx


def test_print_network(edges, queries, target_nodes):
    graph = nx.Graph()
    graph.add_edges_from([(e[0].name, e[1].name) for e in edges])

    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="grey")
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edge_color="grey")

    colors = rand.sample([cname for cname in cnames.keys() if cname.startswith("dark")], len(queries))
    for query, target_node, color in zip(queries, target_nodes, colors):
        nx.draw_networkx_edges(graph, pos, edgelist=query.visited_tree, edge_color=color, width=3)
        nx.draw_networkx_nodes(graph, pos, nodelist=[target_node.name], node_color=color)


def calc_accuracy_per_ttl(queries, query_results):
    assert len(queries) > 0
    acc = np.zeros((len(queries), queries[0].messages[0].ttl))
    for i, query in enumerate(queries):
        if query.candidate_doc == query_results[query.name]:
            acc[i, query.hops_to_reach_candidate_doc:] = 1
    return np.mean(acc, axis=0)


def queue_monitor():
    n_messages = sum([len(node.query_queue) for node in simulation.nodes])
    print(f"{n_messages} message{'' if n_messages == 1 else 's'} circulating")
    return n_messages > 0


rand.seed(1)
np.random.seed(1)
dim = 5000
graph_name = "fb"
ttl = 100
epochs = 200

all_dims = [50, 100, 500, 1000, 5000]
all_accs = []
for dim in all_dims:
    node2vec_params = {
        "dim": dim,
        "l": 3,  # walk length
        "p": 1,
        "q": 0.5,
        "r": 10,  # number of walks per node
        "k": 10  # context size
    }

    print("load graph")
    graph = load_graph(lambda name: HardSumEmbeddingNode(name, dim), graph_name)
    simulation = DecentralizedSimulation(graph)

    print("create and scatter docs")

    node2vec = load_node2vec(graph_name, as_dict=True, normalized=True, params=node2vec_params)
    nodevecs = [node2vec[node.name] for node in graph.nodes]
    # nodevecs = random_unitary(graph.number_of_nodes(), dim)
    docs = []
    for node, vec in zip(graph.nodes, nodevecs):
        doc = Document(f"{node.name}", vec)
        docs.append(doc)
        node.add_doc(doc)

    print("diffuse embeddings")
    simulation.learn_neighbors()

    print("scatter queries")
    # queries = []
    # for i, (node, vec) in enumerate(zip(graph.nodes, nodevecs)):
    #     for start_node in graph.nodes:
    #         query = Query(f"{node.name}", vec)
    #         start_node.add_query(query.spawn(ttl))
    #         queries.append(query)
    #     if i >= 4:
    #         break

    target_nodes = simulation.sample_nodes(1)
    queries = []
    for target_node in target_nodes:
        for node in graph.nodes:
            query = Query(f"{target_node.name}", node2vec[target_node.name])
            node.add_query(query.spawn(ttl))
            queries.append(query)

    # n_samples = 30000
    # start_nodes = rand.choices(simulation.nodes, k=n_samples)
    # target_nodes = rand.choices(simulation.nodes, k=n_samples)
    # queries = []
    # for start_node, target_node in zip(start_nodes, target_nodes):
    #     query = Query(f"{target_node.name}", target_node.embedding)
    #     start_node.add_query(query.spawn(ttl))
    #     queries.append(query)

    query_results = {f"{node.name}": f"{node.name}" for node in graph.nodes}
    nodedict = {node.name: node for node in simulation.nodes}
    docdict = {doc.name: doc for doc in docs}
    querydict = {query.name: query for query in queries}


    print("begin simulation")
    time = simulation.run_queries(epochs, queue_monitor)
    print("Finished search at time", time)

    acc = calc_accuracy_per_ttl(queries, query_results)
    all_accs.append(acc)

plt.figure()
for dim, acc in zip(all_dims, all_accs):
    plt.plot(range(1, len(acc)+1), acc, label=f"dim {dim}")
plt.legend()
plt.grid()
plt.xlabel("TTL (hops)")
plt.ylabel("Accuracy (%)")


# succeeded = [query.messages[0] for query in queries if query.candidate_doc == query.name]
# failed = [query.messages[0] for query in queries if query.candidate_doc != query.name]
#
# # dist = lambda node1, node2: nodedict[node1].embedding @ nodedict[node2].embedding
# dist = lambda node1, node2: np.linalg.norm(nodedict[node1].embedding - nodedict[node2].embedding)
