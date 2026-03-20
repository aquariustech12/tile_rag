from pathlib import Path
from importlib import util
p = Path('~/tile_rag/vama-agent2-fixed.py').expanduser()
spec = util.spec_from_file_location("vama_agent_mod", p)
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
queries = [
  "Pisos para baño color blanco 5 m2",
  "Pisos para cocina 10 m2 gris",
  "Pisos exteriores 20 m2 antideslizante",
  "Mosaico 30x30 blanco",
  "Pisos 60x120 blanco brillo",
  "Pisos madera 20 m2",
  "Pisos para terraza 15 m2",
  "Pisos para baño 3 m2 gris",
  "Pisos 5 m2 beige",
  "Pisos 10 m2 negro"
]
for q in queries:
    matches = mod.find_pisos_from_csv(q)
    print("QUERY:", q)
    if matches:
        for m in matches:
            print("  -", m['codigo'], m['descripcion'], m.get('precio',''))
    else:
        print("  - NO MATCHES")
    print()
