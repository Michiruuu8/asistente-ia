import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

INSTRUCCION_SISTEMA = "Eres un asistente de IA personal, amigable y conciso. Responde en español, de forma breve y natural, como si fueras un asistente estilo Jarvis."

def preguntar_al_llm(mensaje_usuario):
    """Envia un mensaje a Gemini y devuelve su respuesta como texto."""
    respuesta = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=mensaje_usuario,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCCION_SISTEMA,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    return respuesta.text