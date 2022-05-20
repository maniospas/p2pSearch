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

    def __call__(self, inputs, sparse_dim=None):
        sparse_dim = sparse_dim or self.sparse_dim
        transformed = self.lin(inputs)
        topk = torch.topk(transformed, k=sparse_dim, dim=inputs.ndim-1)
        sparse_topk = [SparseVector(indices.numpy(), values, self.lin.out_features)
                       for indices, values in zip(topk.indices.unbind(), topk.values.unbind())]
        return SparseVector.max_pool(sparse_topk)


class UHD(torch.nn.Module):

    def __init__(self, emb_dim, sparse_dim):
        super(UHD, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.model = BertModel.from_pretrained("bert-base-uncased")

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.model_dim = self._get_model_dim()
        self.emb_dim = emb_dim
        self.sparse_dim = sparse_dim

        self.wta = WTA(self.model_dim, emb_dim, sparse_dim)

    def _get_model_dim(self):
        test_input = self.tokenizer("random text", return_tensors="pt", truncation=True)
        test_output = self.model(**test_input).last_hidden_state.detach()
        return test_output.shape[-1]

    def __call__(self, text, sparse_dim=None):
        tokens = self.tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
        tokens_emb = self.model(**tokens).last_hidden_state.squeeze()
        return self.wta(tokens_emb, sparse_dim)


class Scorer:

    def __init__(self, emb_dim, sparse_dim):
        self.emb_dim = emb_dim
        self.sparse_dim = sparse_dim
        self.uhd = UHD(emb_dim, sparse_dim)

    def score(self, query, document):
        qvec = self.uhd(query)
        dvec = self.uhd(document)
        return qvec @ dvec


# indices = torch.tensor([1, 2, 3])
# values = torch.tensor([.6, .7, -.8])
# x = SparseVector(indices.numpy(), values, 10)
# indices = torch.tensor([1, 2, 5])
# values = torch.tensor([.65, .71, -.83])
# y = SparseVector(indices.numpy(), values, 10)
# w = SparseVector.max_pool([x,y])
# print(w)

# wta = WTA(6, 100, 3)
# inputs = torch.randint(0, 100, (4, 6), dtype=torch.float)
# outputs = wta(inputs)

# encoder = UHD(1000, 5)
# text = "Hello, its me"
# enc = encoder(text)
# print(enc)
