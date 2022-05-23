from collections import defaultdict


def unpack(seq):
    return zip(*seq)


def batchify(seq, batch_size):
    start = 0
    while start < len(seq):
        yield seq[start: start + batch_size]
        start += batch_size


def pairs(seq):
    pairs = []
    for i, item in enumerate(seq):
        for other_item in seq[i + 1:]:
            pairs.append((item, other_item))
    return pairs


def triples(qrels):

    qrels_dict = defaultdict(lambda: [])
    for query, doc, rel in qrels:
        qrels_dict[query].append((doc, rel))

    triples = []
    for query in qrels_dict:
        triples.extend([(query, pos_doc, pos_rel, neg_doc, neg_rel)
                        for (pos_doc, pos_rel), (neg_doc, neg_rel) in pairs(qrels_dict[query])])
    return triples

