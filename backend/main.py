from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import predict
from skills import hora as skill_hora

app = FastAPI(title="Asistente IA - Backend")

class Mensaje(BaseModel):
    texto: str

# Aqui mapeamos que intencion activa que skill.
# Las intenciones que no aparecen aqui (conversacion, saludo, busqueda)
# mas adelante las mandaremos a Claude.
SKILLS_DISPONIBLES = {
    "hora": skill_hora.ejecutar,
}

@app.get("/")
def home():
    return {"mensaje": "El backend del asistente esta funcionando"}

@app.post("/chat")
def chat(mensaje: Mensaje):
    intencion, confianza = predict(mensaje.texto)

    if intencion in SKILLS_DISPONIBLES:
        respuesta = SKILLS_DISPONIBLES[intencion]()
    else:
        respuesta = f"(Todavia no tengo una skill para '{intencion}', pronto conectare con Claude para esto)"

    return {
        "texto_recibido": mensaje.texto,
        "intencion": intencion,
        "confianza": round(confianza, 1),
        "respuesta": respuesta
    }