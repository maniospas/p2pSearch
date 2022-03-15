import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv

from data.network import get_ppr_matrix_path
import os


def unitary(vecs):
    return vecs / np.linalg.norm(vecs, keepdims=True, axis=vecs.ndim-1)


def random_unitary(n, dim):
    vecs = np.random.randn(n, dim)
    return unitary(vecs)


def search(nodes, query_embedding):
    return max([doc for node in nodes for doc in node.docs.values()],
               key=lambda doc: np.sum(doc.embedding * query_embedding))


def softmax(arr, axis=0):
    exp = np.exp(arr)
    return exp / exp.sum(axis=axis, keepdims=True)


def powerseries(W, tol=1e-4, max_tries=500):
    n = W.shape[0]
    I = sparse.identity(n)
    S_old = I
    S = S_old
    for i in range(max_tries):

        S = I + W @ S_old
        dS = np.max(np.abs(S - S_old))
        # print(f"iter {i + 1} diff {dS}")
        if dS < tol:
            break
        S_old = S
    return S


def analytic_ppr_from_graph(graph, alpha, symmetric=True):
    adj = nx.adj_matrix(graph).tocsc()
    if alpha > 0.5:
        return power_analytic_ppr(adj, alpha, symmetric).toarray()
    else:
        return exact_analytic_ppr(adj, alpha, symmetric).toarray()


def analytic_ppr(adj, alpha, symmetric=True, _graph_name=None):

    if _graph_name is not None:
        filepath = get_ppr_matrix_path(_graph_name, alpha, symmetric=True)
        if os.path.exists(filepath):
            return np.load(filepath)

    if alpha > 0.5:
        ppr_mat = power_analytic_ppr(adj, alpha, symmetric).toarray()
    else:
        ppr_mat = exact_analytic_ppr(adj, alpha, symmetric).toarray()

    if _graph_name is not None:
        np.save(filepath, ppr_mat)
    return ppr_mat


# no dangling nodes
def power_analytic_ppr(adj, alpha, symmetric=True):
    D = np.array(adj.sum(axis=1)).squeeze()
    if symmetric:
        invsqrtD = sparse.diags(D**-0.5)
        trans_mat = invsqrtD @ adj.transpose() @ invsqrtD
    else:
        invD = sparse.diags(D**-1.0)
        trans_mat = adj.transpose() @ invD

    ppr_mat = alpha * powerseries((1-alpha) * trans_mat)
    return ppr_mat


def exact_analytic_ppr(adj, alpha, symmetric=True):
    n = adj.shape[0]
    I = sparse.identity(n, format="csc")
    D = np.array(adj.sum(axis=1)).squeeze()
    if symmetric:
        invsqrtD = sparse.diags(D ** -0.5)
        trans_mat = invsqrtD @ adj.transpose() @ invsqrtD
    else:
        invD = sparse.diags(D ** -1.0)
        trans_mat = adj.transpose() @ invD
    return alpha * inv(I - (1 - alpha) * trans_mat)


if __name__ == "__main__":

    from loader import load_graph
    from nodes import Node
    import random
    import time
    import os
    import matplotlib.pyplot as plt

    graph_name = "fb"
    graph = load_graph(Node, graph_name)
    n = graph.number_of_nodes()
    pers = np.zeros((n, 50))
    idxs = random.sample(range(n), k=n//5)
    pers[idxs] = np.random.normal(0, 1, (len(idxs), pers.shape[1]))

    alpha_vals = np.arange(0.1, 0.91, 0.1)
    elapsed_power = []
    elapsed_exact = []
    for alpha in alpha_vals:
        print(f"for alpha {alpha}")
        start = time.time()
        ppr1 = power_analytic_ppr(nx.adjacency_matrix(graph), alpha, pers)
        elapsed = time.time() - start
        elapsed_power.append(elapsed)
        print(f"power method {elapsed} secs")

        start = time.time()
        ppr2 = exact_analytic_ppr(nx.adjacency_matrix(graph), alpha, pers)
        elapsed = time.time() - start
        elapsed_exact.append(elapsed)
        print(f"exact method {elapsed} secs")

        print(f"difference {np.max(np.abs(ppr1-ppr2))}")

    plt.figure()
    plt.grid()
    plt.plot(alpha_vals, elapsed_power, label="power")
    plt.plot(alpha_vals, elapsed_exact, label="exact")
    plt.legend()
    plt.xlabel("PPR alpha")
    plt.ylabel("Time (secs)")
    plt.title(f"PPR time analysis for graph {graph_name}")

    imgs_path = os.path.join(os.path.dirname(__file__), "img")
    figsfolder_path = os.path.join(imgs_path, "ppr_delay_analysis")
    if not os.path.exists(figsfolder_path):
        os.mkdir(figsfolder_path)
    fig_path = os.path.join(figsfolder_path, graph_name)
    plt.savefig(fig_path)
