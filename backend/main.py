from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import predict

app = FastAPI(title="Asistente IA - Backend")

# Definimos la forma que debe tener el mensaje que nos manden
class Mensaje(BaseModel):
    texto: str

@app.get("/")
def home():
    return {"mensaje": "El backend del asistente esta funcionando"}

@app.post("/chat")
def chat(mensaje: Mensaje):
    intencion, confianza = predict(mensaje.texto)
    return {
        "texto_recibido": mensaje.texto,
        "intencion": intencion,
        "confianza": round(confianza, 1)
    }