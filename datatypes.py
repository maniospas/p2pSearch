import numpy as np


class Document:
    def __init__(self, name, embedding):
        self.name = name
        self.embedding = embedding


class ExchangedQuery:
    def __init__(self, name, embedding, hops=0, candidate_doc=None, candidate_doc_similarity=-float('inf')):
        self.name = name
        self.embedding = embedding
        self.hops = hops
        self.hops_to_reach_doc = 0
        self.candidate_doc = candidate_doc
        self.candidate_doc_similarity = candidate_doc_similarity

    def send(self):
        return ExchangedQuery(self.name, self.embedding, self.hops+1, self.candidate_doc, self.candidate_doc_similarity)

    def receive(self, other):
        assert self.name == other.name
        if other.candidate_doc_similarity > self.candidate_doc_similarity:
            self.candidate_doc = other.candidate_doc
            self.hops_to_reach_doc = other.hops_to_reach_doc
            self.candidate_doc_similarity = other.candidate_doc_similarity

    def check_now(self, docs):
        for doc in docs:
            score = np.sum(docs[doc].embedding * self.embedding)
            if score > self.candidate_doc_similarity:
                self.candidate_doc_similarity = score
                self.candidate_doc = doc
                self.hops_to_reach_doc = self.hops
