import json
import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vocab import encode
from model.network import IntentClassifier

# --- 1. Cargar el vocabulario y las categorias guardadas ---
with open("model/saved/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open("model/saved/labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

# labels quedo guardado como {"clima": 0, "hora": 1, ...}
# para predecir necesitamos el sentido inverso: {0: "clima", 1: "hora", ...}
id_to_label = {v: k for k, v in labels.items()}

vocab_size = len(vocab)
num_classes = len(labels)

# --- 2. Recrear el modelo con la misma arquitectura y cargar los pesos entrenados ---
model = IntentClassifier(vocab_size=vocab_size, num_classes=num_classes)
model.load_state_dict(torch.load("model/saved/intent_model.pt"))
model.eval()

# --- 3. Funcion para predecir una frase nueva ---
def predict(text):
    encoded = encode(text, vocab)
    input_tensor = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_id = torch.max(probabilities, dim=1)

    intencion = id_to_label[predicted_id.item()]
    confianza = confidence.item() * 100

    return intencion, confianza

# --- 4. Probar con frases nuevas, que el modelo nunca vio ---
if __name__ == "__main__":
    frases_prueba = [
        "esta soleado hoy",
        "que horas son",
        "no olvides comprar pan",
        "pon una cancion de rock",
        "quien invento la bombilla",
        "hola que tal",
        "cuentame algo gracioso",
    ]

    print("--- Probando el modelo con frases nuevas ---\n")
    for frase in frases_prueba:
        intencion, confianza = predict(frase)
        print(f"'{frase}' -> {intencion} ({confianza:.1f}% confianza)")