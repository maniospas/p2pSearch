from datatypes import Document, ExchangedQuery


class BaseNode:
    def __init__(self, name):
        self.name = name
        self.docs = dict() # filled by external code
        self.queries = dict() # filled by external code (may be empty at the beginning)

    def add_doc(self, doc: Document):
        self.docs[doc.name] = doc

    def add_query(self, query: ExchangedQuery):
        self.queries[query.name] = query

    def update(self):
        pass  # nothing to do

    # send all queries to neighbor (selected by external code)
    def send(self, neighbor):
        return [query.send() for query in self.queries.values()]

    # receive queries from neighbor (selected by external code)
    def receive(self, neighbor, message):
        neighbor_queries = message
        for query in neighbor_queries:
            if query.name in self.queries:
                self.queries[query.name].receive(query)
            else:
                self.queries[query.name] = query
        for query in self.queries.values():
            query.check_now(self.docs)
