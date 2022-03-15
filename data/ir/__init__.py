import os
import sys

DATA_DIR = os.path.dirname(__file__)

TYPE2FILENAME_DICT = {
    "q": "queries",
    "query": "queries",
    "queries": "queries",
    "d": "docs",
    "doc": "docs",
    "docs": "docs",
    "document": "docs",
    "documents": "docs",
    "o": "other_docs",
    "other": "other_docs",
    "others": "other_docs",
    "other_docs": "other_docs",
}


def get_dataset_path(dataset):
    return os.path.join(DATA_DIR, dataset)


def get_qrels_path(dataset):
    dset_path = get_dataset_path(dataset)
    return os.path.join(dset_path, "qrels.txt")


def get_texts_path(dataset, type):
    dset_path = get_dataset_path(dataset)
    return os.path.join(dset_path, TYPE2FILENAME_DICT[type]+".txt")


def get_embeddings_path(dataset, type):
    dset_path = get_dataset_path(dataset)
    return os.path.join(dset_path, TYPE2FILENAME_DICT[type]+"_embs.npz")


def get_clusters_path(dataset, n_clusters):
    dset_path = get_dataset_path(dataset)
    return os.path.join(dset_path, "clusters", f"{n_clusters}_clusters.npy")


def download(dataset):

    dset_path = get_dataset_path(dataset)
    script_path = os.path.join(dset_path, "generate_script.py")
    os.system(f"python {script_path} --path {dset_path}")

