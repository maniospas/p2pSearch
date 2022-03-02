import os
import shutil
import zipfile
import requests
import argparse
import numpy as np
import random

from simcse import SimCSE
from io import BytesIO


def write_texts(name, ids, texts, delimiter="\t"):
    filepath = os.path.join(DIRNAME, name+".txt")
    with open(filepath, "w", encoding="utf8") as f:
        for id_, text in zip(ids, texts):
            f.write(f"{id_}{delimiter}{text.replace(delimiter, '')}\n")


def write_arrays(name, ids, embs):
    filepath = os.path.join(DIRNAME, name)
    np.savez(filepath, ids=np.array(ids), embs=np.array(embs))


def write_qrels(qrels, delimiter="\t"):
    filepath = os.path.join(DIRNAME, "qrels.txt")
    with open(filepath, "w", encoding="utf8") as f:
        for que_id, doc_id in qrels.items():
            f.write(f"{que_id}{delimiter}{doc_id}{delimiter}1\n")

argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--modelname",
                       help="name of the SimCSE model for encoding",
                       default="sup-simcse-roberta-base")
argparser.add_argument("-q", "--numberofqueries",
                       help="number of extracted queries",
                       default="1000")
argparser.add_argument("-p", "--path",
                       help="download path",
                       default=os.getcwd())
args = vars(argparser.parse_args())
model_name = args["modelname"]
nq = int(args['numberofqueries'])
DIRNAME = args["path"]


import sys
print(sys.argv)
print(args)
# download raw data

raw_path = os.path.join(DIRNAME, "raw")

if not os.path.exists(raw_path):

    url = "https://data.deepai.org/Stsbenchmark.zip"

    print("downloading dataset")
    res = requests.get(url, allow_redirects=True)

    print("decompressing")
    with zipfile.ZipFile(BytesIO(res.content)) as zipped:
        zipped.extractall()
    shutil.move(os.path.join(DIRNAME, "stsbenchmark"), raw_path)

# gather all sentences from raw data
sentences_path = os.path.join(raw_path, "sentences.txt")
if os.path.exists(sentences_path):
    with open(sentences_path, "r", encoding="utf8") as f:
        sentences = [line.rstrip("\n") for line in f]
else:
    print("extracting sentences")
    sentences = set()
    for mode in ["train", "test", "dev"]:
        with open(os.path.join(raw_path, f"sts-{mode}.csv")) as f:
            for line in f:
                tokens = line.rstrip().replace("\n", " ").split("\t")
                sentences.add(tokens[5])
                sentences.add(tokens[6])
    sentences = list(sentences)
    with open(sentences_path, "w", encoding="utf8") as f:
        for sentence in sentences:
            f.write(f"{sentence}\n")


# encode sentences
embeddings_path = os.path.join(raw_path, "embeddings.npy")
if os.path.exists(embeddings_path):
    embs = np.load(embeddings_path)
else:
    print("loading model")
    model = SimCSE("princeton-nlp/"+model_name)

    print("encoding sentences")
    embs = model.encode(sentences).numpy()

    np.save(embeddings_path, embs)

# extract docs and queries from sentences
print("extracting docs and queries")

dots = embs @ embs.transpose()
sim_idxs = np.argsort(-dots, axis=1)[:, 1]
qrels = dict()
for i, j in enumerate(sim_idxs):
    if dots[i, j] > 0.6 and i not in qrels.values() and j not in qrels.keys():
        qrels[i] = j
qrels = dict(random.sample(list(qrels.items()), min(nq, len(qrels))))
que_idxs = list(qrels.keys())
doc_idxs = list(qrels.values())
other_idxs = list(set(range(len(sentences))).difference(set(que_idxs)).difference(set(doc_idxs)))

doc_counter = 0
que_counter = 0
labels = []
for idx in range(len(sentences)):
    if idx in que_idxs:
        labels.append(f"que{que_counter}")
        que_counter += 1
    else:
        labels.append(f"doc{doc_counter}")
        doc_counter += 1

sentences = np.array(sentences)
labels = np.array(labels, dtype="object")
qrels = {labels[k]: labels[v] for k, v in qrels.items()}
que_ids = labels[que_idxs]
que_texts = sentences[que_idxs]
que_embs = embs[que_idxs]
doc_ids = labels[doc_idxs]
doc_texts = sentences[doc_idxs]
doc_embs = embs[doc_idxs]
other_ids = labels[other_idxs]
other_texts = sentences[other_idxs]
other_embs = embs[other_idxs]

print("validating")

qlabels = que_ids
qvecs = que_embs
dlabels = np.concatenate((doc_ids, other_ids))
dvecs = np.row_stack((doc_embs, other_embs))
sims = np.argmax(qvecs @ dvecs.transpose(), axis=1)
estim_qrels = dict(zip(qlabels, dlabels[sims]))
n_correct = sum([did == qrels[qid] for qid, did in estim_qrels.items()])
print(f"Validated {n_correct}/{len(qlabels)} queries")

print("storing")


write_texts("queries", que_ids, que_texts)
write_arrays("queries_embs", que_ids, que_embs)


write_texts("docs", doc_ids, doc_texts)
write_arrays("docs_embs", doc_ids, doc_embs)


write_texts("other_docs", other_ids, other_texts)
write_arrays("other_docs__embs", other_ids, other_embs)

write_qrels(qrels)

print("finished")
