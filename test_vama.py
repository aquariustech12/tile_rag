import requests
import json
import socket

def test_vama_bridge():
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "qwen2.5:14b",
        "prompt": "Hola, responde con la palabra 'LISTO'",
        "stream": False
    }
    
    print("1. Probando bypass de librería (HTTP puro con requests)...")
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            print(f"✅ CONEXIÓN EXITOSA: {r.json().get('response')}")
        else:
            print(f"❌ ERROR DE OLLAMA (Código {r.status_code}): {r.text}")
    except Exception as e:
        print(f"❌ FALLO DE RED EN PYTHON 3.14: {e}")

    print("\n2. Verificando localhost...")
    try:
        ip = socket.gethostbyname('localhost')
        print(f"Localhost resuelve a: {ip}")
    except Exception as e:
        print(f"❌ Error resolviendo localhost: {e}")

if __name__ == "__main__":
    test_vama_bridge()
