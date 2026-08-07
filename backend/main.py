from fastapi import FastAPI

app = FastAPI(title="Asistente IA - Backend")

@app.get("/")
def home():
    return {"mensaje": "El backend del asistente esta funcionando"}