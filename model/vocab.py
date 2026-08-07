import re
import json

def tokenize(text):
    """Convierte una frase en una lista de palabras limpias."""
    text = text.lower()
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    tokens = text.split()
    return tokens

def build_vocab(intents_dict):
    """Recorre todas las frases del dataset y arma el diccionario palabra -> numero."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for categoria, frases in intents_dict.items():
        for frase in frases:
            for palabra in tokenize(frase):
                if palabra not in vocab:
                    vocab[palabra] = idx
                    idx += 1
    return vocab

def encode(text, vocab, max_len=12):
    """Convierte una frase en una lista de numeros de longitud fija."""
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def build_labels(intents_dict):
    """Asigna un numero a cada categoria (clima=0, hora=1, etc)."""
    labels = {categoria: i for i, categoria in enumerate(intents_dict.keys())}
    return labels