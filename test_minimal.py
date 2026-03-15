# test_minimal.py
import ollama

for host in ['http://127.0.0.1:11434', 'http://localhost:11434', 'http://[::1]:11434']:
    try:
        client = ollama.Client(host=host)
        models = client.list()
        print(f"✅ {host} - OK")
        print(f"   Modelos: {[m['model'] for m in models['models']]}\n")
    except Exception as e:
        print(f"❌ {host} - Error: {e}\n")