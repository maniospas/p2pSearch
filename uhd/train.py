from uhd.model import Scorer
from data.ir.datasets import Cranfield
from random import shuffle
import torch.optim as optim

import cProfile

epochs = 1
emb_dim = 80000
sparse_dim = 80

scorer = Scorer(emb_dim, sparse_dim)
dset = Cranfield()
qrels = list(dset.stream_qrels(as_str=True, binary=False))

optimizer = optim.SGD(scorer.uhd.parameters(), lr=0.001, momentum=0.9)
profile = cProfile.Profile()
profile.enable()
for epoch in range(epochs):
    print(f"EPOCH {epoch+1}/{epochs}")
    shuffle(qrels)
    running_loss = 0.0
    for i, (query, doc, rel) in enumerate(qrels):
        print(f"\r{int(100*i/len(qrels))}% complete", end="")
        optimizer.zero_grad()
        loss = (rel - scorer.score(query, doc))**2
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i == 2:
            break
    print(f"mean loss = {running_loss / len(qrels):.3f}")
profile.disable()
