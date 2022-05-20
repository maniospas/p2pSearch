
if __name__ == "__main__":
    import torch
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    texts = ["why hello there", "fanciful oranges"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    embs = outputs.last_hidden_state