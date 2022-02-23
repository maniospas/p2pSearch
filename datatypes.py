import numpy as np


class Document:
    def __init__(self, name, embedding):
        self.name = name
        self.embedding = embedding

    def __repr__(self):
        return f"{self.__class__.__name__} (\'{self.name}\')"


class Query:
    def __init__(self, name, embedding):
        self.name = name
        self.embedding = embedding
        self.realizations = []

    def register(self, realization):
        self.realizations.append(realization)

    @property
    def candidate_doc(self):
        candidate_doc, max_similarity = None, -float('inf')
        for q in self.realizations:
            if q.candidate_doc_similarity > max_similarity:
                max_similarity = q.candidate_doc_similarity
                candidate_doc = q.candidate_doc
        return candidate_doc

    def __repr__(self):
        return f"{self.__class__.__name__} (\'{self.name}\')"

class MessageQuery:
    def __init__(self, query, ttl):
        self.query = query
        self.ttl = ttl
        self.hops = 0
        self.hops_to_reach_doc = 0
        self.candidate_doc = None
        self.candidate_doc_similarity = -float('inf')
        # self.history = []

        # notify original query so that self can be monitored
        self.query.register(self)

    @property
    def name(self):
        return self.query.name

    @property
    def embedding(self):
        return self.query.embedding

    def is_alive(self):
        return self.hops < self.ttl

    def kill(self):
        pass # TODO notify query

    def clone(self):
        copy = MessageQuery(self.query, self.ttl)
        copy.hops = self.hops
        copy.hops_to_reach_doc = self.hops_to_reach_doc
        copy.candidate_doc = self.candidate_doc
        copy.candidate_doc_similarity = self.candidate_doc_similarity
        return copy

    def send(self):
        self.hops += 1
        # self.history.append((from_node, to_node))
        return self

    def receive(self, other):
        assert self.name == other.name
        if other.candidate_doc_similarity > self.candidate_doc_similarity:
            self.candidate_doc = other.candidate_doc
            self.hops_to_reach_doc = other.hops_to_reach_doc
            self.candidate_doc_similarity = other.candidate_doc_similarity
        self.hops = max(self.hops, other.hops)

    def check_now(self, docs):
        for doc in docs:
            score = np.sum(docs[doc].embedding * self.embedding)
            if score > self.candidate_doc_similarity:
                self.candidate_doc_similarity = score
                self.candidate_doc = doc
                self.hops_to_reach_doc = self.hops

    def __repr__(self):
        return f"{self.__class__.__name__} (\'{self.name}\', {self.ttl-self.hops} hops remaining)"
