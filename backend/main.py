from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import predict
from skills import hora as skill_hora
from backend.gemini_client import preguntar_al_llm

app = FastAPI(title="Asistente IA - Backend")

class Mensaje(BaseModel):
    texto: str

SKILLS_DISPONIBLES = {
    "hora": skill_hora.ejecutar,
}

UMBRAL_CONFIANZA = 60.0

@app.get("/")
def home():
    return {"mensaje": "El backend del asistente esta funcionando"}

@app.post("/chat")
def chat(mensaje: Mensaje):
    intencion, confianza = predict(mensaje.texto)

    if intencion in SKILLS_DISPONIBLES and confianza >= UMBRAL_CONFIANZA:
        respuesta = SKILLS_DISPONIBLES[intencion]()
    else:
        respuesta = preguntar_al_llm(mensaje.texto)

    return {
        "texto_recibido": mensaje.texto,
        "intencion": intencion,
        "confianza": round(confianza, 1),
        "respuesta": respuesta
    }