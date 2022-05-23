from pandas.conftest import narrow_series
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import torch


class SparseVector:

    def __init__(self, indices, values, dim):
        self.dim = dim
        self.value_dict = {k: v for k, v in zip(indices, values)}

    def _vector_view(self, as_type=list):
        indices = self.indices
        values = [self.value_dict[idx] for idx in indices]
        return as_type(indices), as_type(values)

    @property
    def indices(self):
        return sorted(list(self.value_dict.keys()))

    @property
    def values(self):
        return [self.value_dict[idx] for idx in self.indices]

    def __add__(self, other):
        indices = set(self.value_dict.keys()).union(set(other.value_dict.keys()))
        values = [self.value_dict.get(idx, 0) + other.value_dict.get(idx, 0) for idx in indices]
        return SparseVector(indices, values, self.dim)

    def __mul__(self, other):
        indices = set(self.value_dict.keys()).intersection(set(other.value_dict.keys()))
        values = [self.value_dict[idx] * other.value_dict[idx] for idx in indices]
        return SparseVector(indices, values, self.dim)

    def __matmul__(self, other):
        indices = set(self.value_dict.keys()).intersection(set(other.value_dict.keys()))
        return sum([self.value_dict[idx] * other.value_dict[idx] for idx in indices])

    @classmethod
    def max_pool(self, vecs):
        pooled_value_dict = dict()
        for vec in vecs:
            for idx, val in vec.value_dict.items():
                if idx not in pooled_value_dict or (idx in pooled_value_dict and val > pooled_value_dict[idx]):
                    pooled_value_dict[idx] = val
        indices, values = zip(*pooled_value_dict.items())

        return SparseVector(indices, values, vec.dim)

    def to_dense(self, as_type=list):
        return as_type([self.value_dict.get(idx, 0) for idx in range(self.dim)])

    def __repr__(self):
        indices, values = self._vector_view(as_type=list)
        return f"{self.__class__.__name__}(indices={indices}, values={values})"

    def __str__(self):
        indices, values = self._vector_view(as_type=list)
        representation = ", ".join([f"{idx}:{value:.3f}" for idx, value in zip(indices, values)])
        return f"sparse({representation})"


class WTA(torch.nn.Module):

    def __init__(self, input_dim, output_dim, sparse_dim):
        super(WTA, self).__init__()
        self.lin = torch.nn.Linear(input_dim, output_dim)
        self.sparse_dim = sparse_dim

    # works with 3D
    def __call__(self, inputs, sparse_dim=None):

        is_2d = inputs.ndim == 2
        inputs = inputs.unsqueeze(0) if is_2d else inputs

        sparse_dim = sparse_dim or self.sparse_dim
        transformed = self.lin(inputs)
        topk = torch.topk(transformed, k=sparse_dim, dim=inputs.ndim-1)
        pooled = torch.full_like(transformed, -torch.inf)\
            .scatter_(dim=inputs.ndim-1, index=topk.indices, src=topk.values).max(dim=inputs.ndim-2).values
        sparse_vecs = []
        for one_pooled in pooled:
            inds = torch.nonzero(one_pooled > -torch.inf).squeeze()
            sparse_vecs.append(SparseVector(indices=inds.tolist(), values=one_pooled[inds], dim=self.lin.out_features))

        sparse_vecs = sparse_vecs[0] if is_2d else sparse_vecs
        return sparse_vecs

        # sparse_topk = [SparseVector(indices.numpy(), values, self.lin.out_features)
        #                for indices, values in zip(topk.indices.unbind(), topk.values.unbind())]
        # return SparseVector.max_pool(sparse_topk)


class UHD(torch.nn.Module):

    def __init__(self, emb_dim, sparse_dim):
        super(UHD, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        # self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.model_dim = self._get_model_dim()
        self.emb_dim = emb_dim
        self.sparse_dim = sparse_dim

        self.wta = WTA(self.model_dim, emb_dim, sparse_dim)

    def _get_model_dim(self):
        test_input = self.tokenizer("random text", return_tensors="pt", truncation=True)
        test_output = self.model(**test_input).last_hidden_state.detach()
        return test_output.shape[-1]

    def __call__(self, texts, sparse_dim=None):

        is_single_text = isinstance(texts, str)
        texts = [texts] if is_single_text else texts

        tokens = self.tokenizer(texts, return_tensors="pt", padding="longest", truncation=True)
        tokens_emb = self.model(**tokens).last_hidden_state

        output = self.wta(tokens_emb, sparse_dim)
        output = output[0] if is_single_text else output
        return output


class Scorer:

    def __init__(self, emb_dim, sparse_dim):
        self.emb_dim = emb_dim
        self.sparse_dim = sparse_dim
        self.uhd = UHD(emb_dim, sparse_dim)

    def score_one(self, query, document):
        qvec = self.uhd(query)
        dvec = self.uhd(document)
        return qvec @ dvec

    def score_pairs(self, queries, documents):
        qvecs = self.uhd(queries)
        dvecs = self.uhd(documents)
        return torch.cat([(qvec @ dvec).unsqueeze(0) for qvec, dvec in zip(qvecs, dvecs)])


# from torchviz import make_dot
#
# input_emb_dim = 5
# output_emb_dim = 10
# sparse_dim = 3
# n_sentences = 3
# n_tokens = 5
#
# wta = WTA(input_emb_dim, output_emb_dim, sparse_dim)
# x = torch.randn(n_sentences, n_tokens, input_emb_dim, requires_grad=True)
# y = torch.randn(n_sentences, n_tokens, input_emb_dim, requires_grad=True)
# rel = torch.randint(0, 2, (n_sentences,), dtype=torch.float)
# score = torch.cat([(qvec @ dvec).unsqueeze(0) for qvec, dvec in zip(wta(x), wta(y))])
# loss = torch.sum((score - rel)**2)
#
# make_dot(loss, dict(wta.named_parameters()))


# wta = WTA(6, 100, 3)
# # inputs = torch.randint(0, 100, (4, 4, 6), dtype=torch.float)
# inputs = torch.tensor([[1,2,-1,-2], [-1,2,1,-3], [0,1,2,0]],dtype=torch.float, requires_grad=True)
# outputs = wta(inputs, sparse_dim=2, debug=True)
# print(outputs)
# encoder = UHD(1000, 5)


# queries = ["Hello, its me", "hi"]
# docs = ["Nikos", "feasible"]
# scorer = Scorer(100, 20)
#
# preds = scorer.score_pairs(queries, docs)
