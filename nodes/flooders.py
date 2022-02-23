from nodes.base import DocNode


class FlooderNode(DocNode):

    def __init__(self, name):
        super().__init__(name)
        self.seen = set()

    def receive_queries(self, queries, from_node):
        for query in queries:
            if query.name in self.seen:
                query.kill()
                return

            query.check_now(self.docs)

            if query.is_alive():
                self.query_queue[query.name] = query
            else:
                query.kill()

            self.seen.add(query.name)

    def send_queries(self):
        assert all([query.is_alive() for query in self.query_queue.values()]), "queries in query queue should have been alive"
        queries = [query.send() for query in self.query_queue.values()]
        to_send = dict()
        if len(self.neighbors) > 0:
            neighbor_it = iter(self.neighbors)
            to_send[next(neighbor_it)] = queries
            for neighbor in neighbor_it:
                to_send[neighbor] = [query.clone() for query in queries]
        else: # if the node has no known neighbors
            for query in queries:
                query.kill()
        self.query_queue.clear()
        return to_send

from stubs import *

node1 = FlooderNode("paparazzi")
node2 = FlooderNode("womanizer")
node3 = FlooderNode("madonna")

queries, messages = generate_stub_message_queries(3, ttl=10)

node2.receive_embedding(node1.name, node1.send_embedding())
node2.receive_embedding(node3.name, node3.send_embedding())

node2.receive_queries([messages[0]], node1.name)
node2.receive_queries([messages[1]], node1.name)
