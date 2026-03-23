import json
import sqlite3
import os
import glob
from datetime import datetime

DESTINO = "conversaciones_completas.jsonl"

# Lista de fuentes a migrar (solo una vez)
FUENTES = [
    "memoria_v2.json",
    "memoria_v3.json",
    "usuarios.db",
    "vama.db",
    "conversaciones.log",
    "conversaciones_rescatadas.log",
    "backups/conversaciones_20260317.log",
    "backups/conversaciones_20260318.log",
]

def leer_jsonl(archivo):
    """Lee un archivo JSONL y devuelve lista de registros."""
    if not os.path.exists(archivo):
        return []
    registros = []
    with open(archivo, 'r') as f:
        for line in f:
            try:
                registros.append(json.loads(line))
            except:
                continue
    return registros

def escribir_jsonl(registro, destino):
    with open(destino, 'a') as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")

def migrar_desde_json(archivo):
    if not os.path.exists(archivo):
        return
    with open(archivo, 'r') as f:
        data = json.load(f)
    for telefono, user in data.items():
        historial = user.get("historial", [])
        i = 0
        while i < len(historial) - 1:
            user_msg = historial[i]
            bot_msg = historial[i+1]
            if user_msg.get("role") == "user" and bot_msg.get("role") == "assistant":
                registro = {
                    "prompt": user_msg.get("content", ""),
                    "response": bot_msg.get("content", ""),
                    "metadata": {
                        "telefono": telefono,
                        "nombre": user.get("nombre", ""),
                        "timestamp_user": user_msg.get("time", ""),
                        "timestamp_bot": bot_msg.get("time", ""),
                        "estado": user.get("estado", ""),
                        "carrito": user.get("carrito", []),
                        "ultimos_productos": [p.get("codigo") for p in user.get("ultimos_productos", [])],
                        "m2": user.get("m2", 0)
                    }
                }
                escribir_jsonl(registro, DESTINO)
            i += 2

def migrar_desde_sqlite(archivo):
    if not os.path.exists(archivo):
        return
    conn = sqlite3.connect(archivo)
    cur = conn.cursor()
    # Detectamos si tiene tabla 'usuarios' (para vama.db)
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usuarios'")
        if not cur.fetchone():
            # Si no, intentar con tabla 'conversaciones'? (por si hay otro esquema)
            return
        cur.execute("SELECT telefono, nombre, historial FROM usuarios")
    except:
        return
    rows = cur.fetchall()
    for telefono, nombre, historial_json in rows:
        if not historial_json:
            continue
        try:
            historial = json.loads(historial_json)
        except:
            continue
        i = 0
        while i < len(historial) - 1:
            user_msg = historial[i]
            bot_msg = historial[i+1]
            if user_msg.get("role") == "user" and bot_msg.get("role") == "assistant":
                registro = {
                    "prompt": user_msg.get("content", ""),
                    "response": bot_msg.get("content", ""),
                    "metadata": {
                        "telefono": telefono,
                        "nombre": nombre,
                        "timestamp_user": user_msg.get("time", ""),
                        "timestamp_bot": bot_msg.get("time", ""),
                        "estado": "",
                        "carrito": [],
                        "ultimos_productos": [],
                        "m2": 0
                    }
                }
                escribir_jsonl(registro, DESTINO)
            i += 2
    conn.close()

def migrar_desde_log(archivo):
    if not os.path.exists(archivo):
        return
    with open(archivo, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Ver si tiene formato de conversación (campo 'prompt' y 'response')
                if 'prompt' in data and 'response' in data:
                    escribir_jsonl(data, DESTINO)
                # Si es formato antiguo de texto plano, ignoramos por ahora
            except:
                continue

if __name__ == "__main__":
    print("🚀 Iniciando migración única de logs...")
    print(f"📁 Destino: {DESTINO}")

    # Hacemos respaldo del destino actual antes de agregar cosas
    if os.path.exists(DESTINO):
        backup = f"{DESTINO}.backup"
        os.rename(DESTINO, backup)
        print(f"📋 Respaldo de destino actual guardado en {backup}")

    # Crear archivo nuevo vacío
    open(DESTINO, 'w').close()

    # Migrar cada fuente
    for fuente in FUENTES:
        if fuente.endswith('.json') and fuente not in ['conversaciones_completas.jsonl']:
            print(f"📖 Procesando {fuente}...")
            migrar_desde_json(fuente)
        elif fuente.endswith('.db'):
            print(f"📖 Procesando {fuente}...")
            migrar_desde_sqlite(fuente)
        elif fuente.endswith('.log'):
            print(f"📖 Procesando {fuente}...")
            migrar_desde_log(fuente)

    print(f"✅ Migración completada. Todos los registros están en {DESTINO}")

    # Contar líneas
    with open(DESTINO, 'r') as f:
        lineas = sum(1 for _ in f)
    print(f"📊 Total de interacciones: {lineas}")