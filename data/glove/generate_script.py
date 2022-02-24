#import argparse
import os
import numpy as np
import random as rand
import gensim.downloader as api


def normalize(arr, axis=1):
    return arr / np.linalg.norm(arr, axis=axis, keepdims=True)

def dict2arrs(adict):
    keys, values = zip(*adict.items())
    return np.array(keys), np.array(values)


n_queries = 1000
model_name = "glove-wiki-gigaword-50"

print("loading model")
model = api.load(model_name)

words = list(model.key_to_index)
rand.shuffle(words)

print("extract queries and documents")
que_words = set()
doc_words = set()
qrels = {}
for word in words:
    similar_word, score = model.most_similar(word, topn=1)[0]
    if score > 0.6 and word not in doc_words and similar_word not in que_words:
        que_words.add(word)
        doc_words.add(similar_word) # some queries may be similar to the same doc
        qrels[word] = similar_word
    if len(que_words) >= n_queries:
        break
other_words = set(words).difference(doc_words).difference(que_words)
que_words, doc_words, other_words = list(que_words), list(doc_words), list(other_words)

# create ids
que2id = {que_word: f"que{i}" for i, que_word in enumerate(que_words)}
doc2id = {doc_word: f"doc{i}" for i, doc_word in enumerate(doc_words)}
other2id = {other_word: f"doc{len(doc2id)+i}" for i, other_word in enumerate(other_words)}
qrels = {que2id[que_word]: doc2id[doc_word] for que_word, doc_word in qrels.items()}


print("validate")
qwords, qids = dict2arrs(que2id)
qvecs = np.array([normalize(model[word], axis=0) for word in qwords])
docwords, docids = dict2arrs(doc2id)
docvecs = np.array([normalize(model[word], axis=0) for word in docwords])
otherwords, otherids = dict2arrs(other2id)
othervecs = np.array([normalize(model[word], axis=0) for word in otherwords])
dids = np.concatenate([docids, otherids])
dvecs = np.row_stack([docvecs, othervecs])
scores = qvecs @ dvecs.transpose()
correctdocs = np.argmax(scores, axis=1)
validation_qrels = {qids[qindex]: dids[dindex] for qindex, dindex in enumerate(correctdocs)}

n_correct = np.sum([qrels[qid] == validation_qrels[qid] for qid in qids])
print(f"--> {n_correct}/{len(qrels)} validated")

# store
print("storing")
DIRPATH = os.path.dirname(__file__)
for data, label in zip([que2id, doc2id, other2id], ["queries", "docs", "other_docs"]):
    filepath = os.path.join(DIRPATH, f"{label}.txt")
    with open(filepath, "w", encoding="utf8") as f:
        for item, id_ in data.items():
            item = item.replace('\t', ' ')
            f.write(f"{id_}\t{item}\n")

    filepath = os.path.join(DIRPATH, f"{label}_embs")
    items = list(data)
    ids = [data[item] for item in items]
    np.savez(filepath, ids=np.array(ids), embs=normalize(model[items]))

filepath = os.path.join(DIRPATH, f"qrels.txt")
with open(filepath, "w", encoding="utf8") as f:
    for que_id, doc_id in qrels.items():
        f.write(f"{que_id}\t{doc_id}\t{1}\n")

print("finished!")
