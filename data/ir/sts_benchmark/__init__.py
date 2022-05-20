import os
#from simcse import SimCSE

DIRNAME = os.path.join(os.path.dirname(__file__), "data")


def load_sentences():
    sentences = []
    with open(os.path.join(DIRNAME, "sentences.txt"), encoding="utf8") as f:
        for line in f:
            sentences.append(line.rstrip())
    return sentences


def load_embeddings(model_name):
    filename = "sentences_" + model_name + ".npy"
    return np.load(os.path.join(DIRNAME, filename))

    


