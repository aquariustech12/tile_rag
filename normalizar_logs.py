import json
import os

def normalizar():
    entrada = "conversaciones_completas.jsonl"
    salida = "conversaciones_unificadas.jsonl"
    cont = 0

    with open(entrada, "r") as f_in, open(salida, "w") as f_out:
        for linea in f_in:
            linea = linea.strip()
            if not linea:
                continue
            try:
                data = json.loads(linea)
            except:
                continue

            # Si ya está en formato nuevo, la escribimos tal cual
            if "timestamp" in data and "mensaje_usuario" in data:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                cont += 1
                continue

            # Si es formato antiguo (prompt/response)
            if "prompt" in data and "response" in data:
                # Extraer metadata
                meta = data.get("metadata", {})
                nuevo_registro = {
                    "timestamp": meta.get("timestamp_bot", ""),
                    "telefono": meta.get("telefono", ""),
                    "nombre": meta.get("nombre", ""),
                    "mensaje_usuario": data.get("prompt", ""),
                    "respuesta_bot": data.get("response", ""),
                    "estado": meta.get("estado", ""),
                    "carrito": meta.get("carrito", []),
                    "ultimos_productos": meta.get("ultimos_productos", []),
                    "m2": meta.get("m2", 0)
                }
                # Si timestamp está vacío, intentar usar timestamp_user
                if not nuevo_registro["timestamp"]:
                    nuevo_registro["timestamp"] = meta.get("timestamp_user", "")
                f_out.write(json.dumps(nuevo_registro, ensure_ascii=False) + "\n")
                cont += 1
                continue

            # Si no se reconoce, se ignora
            print(f"⚠️ Línea no reconocida: {linea[:100]}...")

    print(f"✅ Normalización completada. {cont} registros guardados en {salida}")

if __name__ == "__main__":
    normalizar()