from .ppr import PPRNode
from datatypes import Document, ExchangedQuery


class FloodNode(PPRNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docs = dict() # filled by external code
        self.queries = dict() # filled by external code (may be empty at the beginning)

    def add_doc(self, doc: Document):
        self.docs[doc.name] = doc

    def add_query(self, query: ExchangedQuery):
        self.queries[query.name] = query

    def update(self):
        personalization = 0
        for doc in self.docs.values():
            personalization = personalization + doc.embedding / len(self.docs)
        self.set_personalization(personalization)
        super().update()

    def send(self, neighbor):
        return super().send(neighbor), [query.send() for query in self.queries.values()]

    def receive(self, neighbor, message):
        neighbor_embedding, neighbor_queries = message
        for query in neighbor_queries:
            if query.name in self.queries:
                self.queries[query.name].receive(query)
            else:
                self.queries[query.name] = query
        for query in self.queries.values():
            query.check_now(self.docs)
        super().receive(neighbor, neighbor_embedding)
