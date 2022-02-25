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
        self.messages = []

    def register(self, message):
        self.messages.append(message)

    @property
    def candidate_doc(self):
        candidate_doc, max_similarity = None, -float('inf')
        for q in self.messages:
            if q.candidate_doc_similarity > max_similarity:
                max_similarity = q.candidate_doc_similarity
                candidate_doc = q.candidate_doc
        return candidate_doc

    @property
    def hops_to_reach_candidate_doc(self):
        candidate_hops, max_similarity = None, -float('inf')
        for q in self.messages:
            if q.candidate_doc_similarity > max_similarity:
                max_similarity = q.candidate_doc_similarity
                candidate_hops = q.hops_to_reach_doc
        return candidate_hops

    @property
    def visited_tree(self):
        tree = set()
        for message in self.messages:
            tree.update(set(message.visited_edges))
        return list(tree)

    def __repr__(self):
        return f"{self.__class__.__name__} (\'{self.name}\')"


class MessageQuery:

    counter = 0

    def __init__(self, query, ttl, name=None):
        if name is None:
            name = f"qm{self.__class__.counter}({query.name})"
            self.__class__.counter += 1
        self.name = name
        # refers to all messages cloned from the same initial message query
        # messages added to different nodes will have diffrent message_names EVEN if they point to the same query obj

        self.query = query
        self.ttl = ttl
        self.hops = 0
        self.hops_to_reach_doc = 0
        self.candidate_doc = None
        self.candidate_doc_similarity = -float('inf')
        self.visited_edges = []

        # notify original query so that self can be monitored
        self.query.register(self)

    @property
    def visited_nodes(self):
        if len(self.visited_edges) == 0:
            return []
        nodes = [edge[0] for edge in self.visited_edges]
        nodes.append(self.visited_edges[-1][1])
        return nodes

    @property
    def query_name(self):
        return self.query.name

    @property
    def embedding(self):
        return self.query.embedding

    def is_alive(self):
        return self.hops < self.ttl

    def kill(self, at_node, reason=""):
        print(f"Query {self.query.name} died at node {at_node.name} because {reason}")
         # TODO notify query

    def clone(self):
        copy = MessageQuery(self.query, self.ttl, name=self.name)
        copy.hops = self.hops
        copy.hops_to_reach_doc = self.hops_to_reach_doc
        copy.candidate_doc = self.candidate_doc
        copy.candidate_doc_similarity = self.candidate_doc_similarity
        return copy

    def send(self, from_node, to_node):
        self.hops += 1
        self.visited_edges.append((from_node.name, to_node.name))
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
