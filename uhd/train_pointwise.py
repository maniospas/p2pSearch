import torch
import cProfile
from random import shuffle

import data.ir.datasets as dsets
from uhd.model import Scorer
from uhd.utils import batchify, unpack


def mse_loss(preds, labels):
    return torch.mean(preds - labels)**2



emb_dim = 80000
sparse_dim = 80
scorer = Scorer(emb_dim, sparse_dim)

dset = dsets.Antique(mode="train")
qrels = list(dset.stream_qrels(as_str=True, binary=False))[:12]

epochs = 100
batch_size = 3
optimizer = torch.optim.Adam(scorer.uhd.parameters(), lr=0.001)
loss_fn = mse_loss

profile = cProfile.Profile()
profile.enable()

loss_history = []
best_state_dict = None
best_loss = 0.0

for epoch in range(epochs):

    shuffle(qrels)
    running_loss = 0.0
    for i, batch in enumerate(batchify(qrels, batch_size)):

        # print(f"\rbatch {i} ({int(i * batch_size / len(triples) * 100)}% complete)", end="")

        queries, docs, rels = unpack(batch)

        optimizer.zero_grad()
        loss = loss_fn(torch.tensor(rels, dtype=torch.float), scorer.score_pairs(queries, docs))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    mean_epoch_loss = running_loss / batch_size
    print(f"EPOCH {epoch + 1}/{epochs} complete, mean loss = {mean_epoch_loss:.3f}")
    loss_history.append(mean_epoch_loss)

    if mean_epoch_loss < best_loss:
        best_loss = mean_epoch_loss
        best_state_dict = scorer.uhd.state_dict()

    if mean_epoch_loss < 0.2:
        break

profile.disable()
