#!/usr/bin/env bash
# apply_fix.sh
# Uso: ./apply_fix.sh [ruta_al_archivo]
# Por defecto actúa sobre ./vama-agent.py

set -e

TARGET="${1:-vama-agent.py}"

if [ ! -f "$TARGET" ]; then
  echo "Archivo no encontrado: $TARGET"
  exit 1
fi

TS=$(date +%s)
BACKUP="${TARGET}.bak.${TS}"
cp "$TARGET" "$BACKUP"
echo "Backup creado: $BACKUP"

python3 - <<'PY'
import re,sys,io
fn = sys.argv[1]
with open(fn, 'r', encoding='utf-8') as f:
    src = f.read()

orig = src

# 1) Insertar normalización justo después de "msg_low = msg_limpio.lower()"
pattern_msglow = r"(msg_low\s*=\s*msg_limpio\.lower\(\))"
norm_block = r"""\1

    # --- Normalización simple y corrección de typos
    replacements = {
        "pisis": "pisos",
        "pisis.": "pisos",
        "clor": "color",
        "clor.": "color",
        "colr": "color",
        "colro": "color",
        "azúl": "azul"
    }
    for a, b in replacements.items():
        msg_low = msg_low.replace(a, b)
"""
# Only insert if not already present
if "Normalización simple y corrección de typos" not in src:
    src = re.sub(pattern_msglow, norm_block, src, count=1, flags=re.M)
    print("-> Normalización insertada")
else:
    print("-> Normalización ya presente, se omite inserción")

# 2) Reemplazar la definición de es_compra por versión más estricta
# Buscamos la línea que define es_compra con any(w in msg_low for w in [
pattern_escompra = re.compile(r"es_compra\s*=\s*any\(w in msg_low for w in 

\[([^\]

]+)\]

\)", re.M)
new_escompra = r"""# --- Mejorar detección de intención de compra: distinguir "quiero opciones" (explorar) de "quiero" (comprar)
    explorar_patterns = re.compile(r'\b(opciones|informaci[oó]n|mostrar|ver|ver opciones|qué tienen|qué hay)\b', re.I)
    compra_verbs = ['me llevo','me lo llevo','lo quiero','comprar','comprame','comprarme','agrega','añade','agregar','pon','incluye','cotiza','cotizar','dame','deme','selecciona','escojo','elijo','prefiero','me quedo','llevo']
    es_compra = any(w in msg_low for w in compra_verbs) and not explorar_patterns.search(msg_low)
"""
if pattern_escompra.search(src):
    src = pattern_escompra.sub(new_escompra, src, count=1)
    print("-> Reemplazada la detección de es_compra")
else:
    # fallback: try to replace a common variant
    if "es_compra = any(w in msg_low for w in [" in src:
        src = src.replace("es_compra = any(w in msg_low for w in [", new_escompra)
        print("-> Reemplazo alternativo de es_compra aplicado")
    else:
        print("-> No se encontró la definición original de es_compra; omitiendo reemplazo")

# 3) Guardar cambios si hubo modificaciones
if src != orig:
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(src)
    print(f"Archivo modificado y guardado: {fn}")
else:
    print("No se realizaron cambios en el archivo.")

# 4) Mostrar un pequeño diff (líneas cambiadas)
import difflib
diff = difflib.unified_diff(orig.splitlines(), src.splitlines(), lineterm='')
for i, line in enumerate(diff):
    if i < 200:
        print(line)
    else:
        break
PY "$TARGET"

echo "Listo. Revisa el archivo modificado y reinicia tu servicio para aplicar cambios."
echo "Si quieres, puedo generar también el diff completo o un script que reinicie el servicio automáticamente."

