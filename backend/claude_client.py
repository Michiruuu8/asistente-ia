import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def preguntar_a_claude(mensaje_usuario):
    """Envia un mensaje a Claude y devuelve su respuesta como texto."""
    respuesta = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system="Eres un asistente de IA personal, amigable y conciso. Responde en español, de forma breve y natural, como si fueras un asistente estilo Jarvis.",
        messages=[
            {"role": "user", "content": mensaje_usuario}
        ]
    )
    return respuesta.content[0].text