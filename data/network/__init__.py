import requests
import gzip
import os
import networkx as nx

from dirs import DATA_DIR

METADATA = {
    "gnutella": {"url": "https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz", "delimiter": "\t"},
    "fb": {"url": "https://snap.stanford.edu/data/facebook_combined.txt.gz", "delimiter": " "},
    "internet": {"url": "https://snap.stanford.edu/data/as20000102.txt.gz", "delimiter": "\t"},
    "erdos": {"n": 50, "p": 0.2}
}

COMMON_DELIMITER = ";"


def download(dataset, filepath):

    # flexible to add datasets with different downloading procedure
    if dataset in ["gnutella", "fb", "internet"]:
        url = METADATA[dataset]["url"]
        print(f"* downloading {dataset} network dataset from {url}")
        res = requests.get(url, allow_redirects=True)
        print(f"** decompressing")
        data = gzip.decompress(res.content)
        print(f"*** transforming to common format")
        with open(filepath, 'wb') as f:
            f.write(data)

        f = open(filepath, "r", encoding="utf8")
        lines = f.readlines()
        f.close()

        with open(filepath, 'w') as f:
            for line in lines:
                if line.startswith("#"):
                    continue
                f.write(line.replace(METADATA[dataset]["delimiter"], COMMON_DELIMITER))
        print(f"**** done")

    elif dataset == "erdos":
        n, p = METADATA[dataset]["n"], METADATA[dataset]["p"]
        g = nx.gnp_random_graph(n, p)
        while not nx.is_connected(g):
            p = min(1, 1.01*p)
            g = nx.gnp_random_graph(n, p)

        with open(filepath, 'w') as f:
            for e in g.edges:
                f.write(f"{e[0]}{COMMON_DELIMITER}{e[1]}\n")

    else:
        raise Exception(f"unknown dataset \'{dataset}\', known datasets: {list(METADATA)}")


def get_filepath(dataset):
    network_dir = os.path.join(DATA_DIR, "network")
    if not os.path.exists(network_dir):
        os.mkdir(network_dir)

    filepath = os.path.join(network_dir, f"{dataset}_edgelist.csv")
    if not os.path.exists(filepath):
        download(dataset, filepath)
    return filepath

