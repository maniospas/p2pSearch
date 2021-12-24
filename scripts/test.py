from loader import *

dset = "antique_test_top10"

results = load_query_results(dset)
doc_embs = load_embeddings(dset, "docs")
que_embs = load_embeddings(dset, "queries")

