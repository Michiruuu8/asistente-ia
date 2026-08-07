import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vocab import build_vocab, build_labels, encode, tokenize
from model.network import IntentClassifier

# --- 1. Cargar el dataset ---
with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

vocab = build_vocab(intents)
labels = build_labels(intents)
num_classes = len(labels)
vocab_size = len(vocab)

print(f"Vocabulario: {vocab_size} palabras")
print(f"Categorias: {list(labels.keys())}")

# --- 2. Preparar los datos en formato PyTorch ---
class IntentDataset(Dataset):
    def __init__(self, intents_dict, vocab, labels):
        self.samples = []
        for categoria, frases in intents_dict.items():
            label_id = labels[categoria]
            for frase in frases:
                encoded = encode(frase, vocab)
                self.samples.append((encoded, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded, label_id = self.samples[idx]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)

dataset = IntentDataset(intents, vocab, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- 3. Crear el modelo ---
model = IntentClassifier(vocab_size=vocab_size, num_classes=num_classes)

# --- 4. Definir como medir el error y como ajustar los pesos ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 5. Ciclo de entrenamiento ---
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    if (epoch + 1) % 10 == 0:
        accuracy = 100 * correct / total
        print(f"Epoca {epoch+1} - loss: {total_loss:.4f} - accuracy: {accuracy:.2f}%")

# --- 6. Guardar el modelo entrenado ---
os.makedirs("model/saved", exist_ok=True)
torch.save(model.state_dict(), "model/saved/intent_model.pt")

with open("model/saved/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

with open("model/saved/labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print("\nModelo guardado en model/saved/")