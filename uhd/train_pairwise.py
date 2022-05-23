import torch
import cProfile
from random import shuffle

import data.ir.datasets as dsets
from uhd.model import Scorer, UHD
from uhd.utils import batchify, unpack, triples


emb_dim = 50000
sparse_dim = 80
uhd = UHD(emb_dim, sparse_dim)
# scorer = Scorer(emb_dim, sparse_dim)

dset = dsets.Antique(mode="train")
triples = triples(dset.stream_qrels(as_str=True, binary=False))[:20]


epochs = 20
batch_size = 3
optimizer = torch.optim.Adam(uhd.parameters(), lr=0.001)
relu = torch.nn.ReLU()

profile = cProfile.Profile()
profile.enable()

loss_history = []
best_state_dict = None
best_loss = float('inf')

for epoch in range(epochs):

    shuffle(triples)
    running_loss = 0.0
    for i, batch in enumerate(batchify(triples, batch_size)):

        queries, pos_docs, pos_rels, neg_docs, neg_rels = unpack(batch)

        optimizer.zero_grad()
        qvecs = uhd(queries)
        pos_dvecs = uhd(pos_docs)
        neg_dvecs = uhd(neg_docs)

        pos_scores = torch.cat([torch.as_tensor(qvec @ dvec).unsqueeze(0) for qvec, dvec in zip(qvecs, pos_dvecs)])
        neg_scores = torch.cat([torch.as_tensor(qvec @ dvec).unsqueeze(0) for qvec, dvec in zip(qvecs, neg_dvecs)])
        margins = torch.tensor(pos_rels, dtype=torch.float) - torch.tensor(neg_rels, dtype=torch.float)
        loss = torch.mean(relu(margins - pos_scores + neg_scores))

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    mean_epoch_loss = running_loss / batch_size
    print(f"EPOCH {epoch + 1}/{epochs} complete, mean loss = {mean_epoch_loss:.3f}")
    loss_history.append(mean_epoch_loss)

    if mean_epoch_loss < best_loss:
        best_loss = mean_epoch_loss
        best_state_dict = uhd.state_dict()

    if mean_epoch_loss < 0.2:
        break


profile.disable()

uhd.load_state_dict(best_state_dict)