import torch
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=32, hidden_dim=64):
        super(IntentClassifier, self).__init__()

        # Convierte cada palabra (numero) en un vector de embed_dim numeros
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Primera capa densa: reduce/transforma la representacion de la frase
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Segunda capa: produce un puntaje por cada categoria posible
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x tiene forma [batch, longitud_frase] -> lista de numeros de palabras

        embedded = self.embedding(x)          # [batch, longitud_frase, embed_dim]
        pooled = embedded.mean(dim=1)         # promedio de las palabras -> [batch, embed_dim]

        hidden = self.fc1(pooled)             # [batch, hidden_dim]
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)

        output = self.fc2(hidden)             # [batch, num_classes]
        return output