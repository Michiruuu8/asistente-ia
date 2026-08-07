from datetime import datetime

def ejecutar():
    """Devuelve la hora actual del sistema, formateada de forma legible."""
    ahora = datetime.now()
    hora_formateada = ahora.strftime("%H:%M")
    return f"Son las {hora_formateada}."