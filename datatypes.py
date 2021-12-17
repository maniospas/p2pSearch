import numpy as np

INT_INF = 10000


class Document:
    def __init__(self, name, embedding):
        self.name = name
        self.embedding = embedding

    def __repr__(self):
        return str(self.name)


class Ranking:

    @classmethod
    def empty(cls, capacity=INT_INF):
        return Ranking([], [], capacity)

    def __init__(self, items, scores, capacity=INT_INF):
        assert len(items) == len(scores)
        self.capacity = capacity
        self.ranking = sorted(zip(items, scores), key=lambda tup: tup[1], reverse=True)[:capacity]

    @property
    def items(self):
        return [item for item, _ in self.ranking]

    @property
    def scores(self):
        return [score for _, score in self.ranking]

    def merge(self, other, capacity=None):
        merged_dict = dict(self.ranking)
        for item, score in other.ranking:
            stored_score = merged_dict.setdefault(item, score)
            if score > stored_score:
                merged_dict[item] = score
        return Ranking(merged_dict.keys(), merged_dict.values(), capacity or self.capacity)

    def __getitem__(self, pos):
        return self.ranking[pos]

    def __str__(self):
        return "\n".join([f"{rank}. {str(item)} (score: {score:.3f})"
                          for rank, (item, score) in enumerate(self.ranking, 1)])

    def __contains__(self, item):
        return item in self.items

    def __iter__(self):
        for item, score in self.ranking:
            yield item, score

    def __len__(self):
        return len(self.ranking)

    def __repr__(self):
        return str(self)


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


class TTLQuery:
    def __init__(self, name, embedding, ttl, capacity, candidate_docs=None):
        self.name = name
        self.embedding = embedding
        self.ttl = ttl
        self.capacity = capacity
        self.candidate_docs = candidate_docs or Ranking.empty(capacity)
        self.history = []

    def is_alive(self):
        return self.ttl > 0

    def send(self, edge):
        if self.ttl > 0:
            self.ttl -= 1
        self.history.append(edge)

    def kill(self):
        self.ttl = 0

    def receive(self, other):
        assert self.name == other.name
        self.candidate_docs = self.candidate_docs.merge(other.candidate_docs)

    def check_now(self, docs):
        # if len(docs) > 0:
        #     breakpoint()
        scores = [np.sum(doc.embedding * self.embedding) for doc in docs.values()]
        ranking = Ranking(docs, scores)
        self.candidate_docs = self.candidate_docs.merge(ranking)

    def __str__(self):
        return f"{self.__class__.__name__} {self.name}, ttl: {self.ttl}, docs: {self.candidate_docs}"

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name}, ttl: {self.ttl}, docs: {self.candidate_docs}"


