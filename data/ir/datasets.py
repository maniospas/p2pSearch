import ir_datasets
from collections import defaultdict


class TextRecord:
    def __init__(self, rec_id, rec_text):
        self.id = rec_id
        self.text = rec_text

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"{self.__class__.__name__}(id: {self.id}, text: {self.text})"

    def __eq__(self, other):
        return self.id == other.id and self.text == other.text

    def __hash__(self):
        return hash(self.id) + hash(self.text)


class Cranfield:

    def __init__(self):
        self.dset = ir_datasets.load("cranfield")

        self.query_dict = {query.query_id: query for query in self.dset.queries_iter()}
        self.doc_store = self.dset.docs_store()

    def stream_queries(self, as_str=False):
        for query in self.dset.queries_iter():
            yield query.text if as_str else TextRecord(query.query_id, query.text)

    def stream_documents(self, as_str=False):
        for doc in self.dset.queries_iter():
            yield doc.text if as_str else TextRecord(doc.doc_id, doc.text)

    def stream_qrels(self, as_str=True, binary=True):
        for qrel in self.dset.qrels_iter():
            if qrel.query_id not in self.query_dict:
                continue
            query_obj = self.query_dict[qrel.query_id]
            query = query_obj.text if as_str else TextRecord(query_obj.query_id, query_obj.text)
            doc_obj = self.doc_store.get(qrel.doc_id)
            doc = doc_obj.text if as_str else TextRecord(doc_obj.doc_id, doc.text)
            rel = qrel.relevance
            if binary:
                rel = 1 if qrel.relevance >= 2 else 0
            yield (query, doc, rel)

