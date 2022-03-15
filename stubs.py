import numpy as np

from datatypes import Document, Query, MessageQuery

def generate_stub_documents(k):
    return [Document(f"doc{i}", np.random.randn(0, 1, 768)) for i in range(k)]

def generate_stub_queries(k):
    return [Query(f"que{i}", np.random.randn(0, 1, 768)) for i in range(k)]

def generate_stub_message_queries(k, ttl=10):
    queries = generate_stub_queries(k)
    return queries, [MessageQuery(query, ttl) for query in queries]


class StubNode:
    def __init__(self, name):
        pass