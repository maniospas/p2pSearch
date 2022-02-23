from abc import ABC, abstractmethod
from datatypes import Document, MessageQuery


class PPRNode:
    def __init__(self, name):
        self.name = name
        self.neighbors = dict()
        self.personalization = 0
        self.embedding = self.personalization

    def set_personalization(self, personalization):
        self.personalization = personalization
        self.embedding = personalization # also resets embedding

    def update(self):
        if len(self.neighbors) == 0:
            return
        embedding = 0
        for neighbor_embedding in self.neighbors.values():
            embedding = embedding + neighbor_embedding
        self.embedding = embedding / len(self.neighbors)**0.5 * 0.9 + self.personalization * 0.1

    def receive_embedding(self, neighbor, neighbor_embedding):
        self.neighbors[neighbor] = neighbor_embedding
        self.update()

    def send_embedding(self):
        return self.embedding / max(1, len(self.neighbors)) ** 0.5


class DocNode(PPRNode, ABC):

    def __init__(self, name):
        super(DocNode, self).__init__(name)
        self.docs = dict()
        self.query_queue = dict()

    def add_doc(self, doc: Document):
        self.docs[doc.name] = doc

    def add_query(self, query: MessageQuery):
        assert query.ttl >= 0, f"{query}, ttl should be >= 0"
        query.check_now()
        if query.is_alive():
            if query.name in self.query_queue:
                self.query_queue[query.name].receive(query)
                query.kill()
            else:
                self.query_queue[query.name] = query
        else: # if ttl was initially 0
            query.kill()

    def has_queries_to_send(self):
        return len(self.query_queue) > 0

    @abstractmethod
    def send_queries(self):
        pass

    @abstractmethod
    def receive_queries(self, queries, from_neighbor):
        pass
