from .ppr import PPRNode


class FloodNode(PPRNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docs = dict()
        self.queries = dict()
        self.query_selected_neighbor = dict()
        self.query_sent_to_neighbors = dict()
        self.query_received_by = dict()
        self.reselect_sender = 1

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
