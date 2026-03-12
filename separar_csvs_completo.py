#!/usr/bin/env python3
"""
separar_csvs_completo.py - Separa TODOS los CSVs en categorías específicas
Los archivos están en data/ y los nuevos se guardan también en data/
"""
import pandas as pd
import os
import shutil

print("🔧 Separando CSVs automáticamente...")
data_dir = "data"

# 1. RESPALDO - Crear carpeta backup
backup_dir = os.path.join(data_dir, "backup_originales")
os.makedirs(backup_dir, exist_ok=True)
print(f"📦 Creando backup en {backup_dir}/")

# 2. PROCESAR GRIFERIA.CSV
print("\n🔨 Procesando griferia.csv...")
griferia_path = os.path.join(data_dir, "griferia.csv")
if os.path.exists(griferia_path):
    # Backup
    shutil.copy2(griferia_path, os.path.join(backup_dir, "griferia.csv"))
    
    df = pd.read_csv(griferia_path)
    print(f"   {len(df)} productos")
    
    # Palabras clave
    lavabo_keywords = ['lavabo', 'lavamanos', 'lava b']
    sanitario_keywords = ['wc', 'inodoro', 'taza', 'sanitario', 'one piece', 'tanque', 'pedestal']
    mueble_keywords = ['mueble', 'gabinete', 'cabinet']
    grifo_keywords = ['monomando', 'grifo', 'mezcladora', 'llave', 'regadera']
    
    desc_lower = df['Descripcion'].str.lower().fillna('')
    
    # Clasificar
    es_lavabo = desc_lower.str.contains('|'.join(lavabo_keywords), na=False, regex=True)
    es_sanitario = desc_lower.str.contains('|'.join(sanitario_keywords), na=False, regex=True)
    es_mueble = desc_lower.str.contains('|'.join(mueble_keywords), na=False, regex=True)
    es_grifo = desc_lower.str.contains('|'.join(grifo_keywords), na=False, regex=True)
    
    # Separar (sin duplicados)
    df_lavabos = df[es_lavabo].copy()
    df_sanitarios = df[es_sanitario & ~es_lavabo].copy()
    df_muebles_grif = df[es_mueble & ~es_lavabo & ~es_sanitario].copy()
    df_griferia = df[es_grifo & ~es_lavabo & ~es_sanitario & ~es_mueble].copy()
    
    # Guardar
    df_lavabos.to_csv(os.path.join(data_dir, "lavabos.csv"), index=False)
    df_sanitarios.to_csv(os.path.join(data_dir, "sanitarios.csv"), index=False)
    df_muebles_grif.to_csv(os.path.join(data_dir, "muebles_griferia.csv"), index=False)
    df_griferia.to_csv(os.path.join(data_dir, "griferia_pura.csv"), index=False)
    
    print(f"   ✅ Creados:")
    print(f"      - lavabos.csv: {len(df_lavabos)}")
    print(f"      - sanitarios.csv: {len(df_sanitarios)}")
    print(f"      - muebles_griferia.csv: {len(df_muebles_grif)}")
    print(f"      - griferia_pura.csv: {len(df_griferia)}")
    
    # Renombrar original
    os.rename(griferia_path, os.path.join(data_dir, "griferia_original.csv"))
else:
    print("   ⏭️  No encontrado")

# 3. PROCESAR OTRAS.CSV (de aquí salen más muebles, tinacos, etc)
print("\n🔨 Procesando otras.csv...")
otras_path = os.path.join(data_dir, "otras.csv")
if os.path.exists(otras_path):
    # Backup
    shutil.copy2(otras_path, os.path.join(backup_dir, "otras.csv"))
    
    df = pd.read_csv(otras_path)
    print(f"   {len(df)} productos")
    
    # Palabras clave para otras categorías
    tinaco_keywords = ['tinaco', 'cisterna']
    espejo_keywords = ['espejo']
    tarja_keywords = ['tarja', 'fregadero', 'lavaplatos']
    mueble_keywords = ['mueble', 'gabinete', 'cabinet']
    herramienta_keywords = ['herramienta', 'redtools']
    
    desc_lower = df['Descripcion'].str.lower().fillna('')
    cat_lower = df['Categoría'].str.lower().fillna('')
    
    # Combinar búsqueda en Descripción y Categoría
    es_tinaco = (desc_lower.str.contains('|'.join(tinaco_keywords), na=False, regex=True)) | (cat_lower.str.contains('tinaco', na=False))
    es_espejo = desc_lower.str.contains('|'.join(espejo_keywords), na=False, regex=True) | (cat_lower.str.contains('espejo', na=False))
    es_tarja = desc_lower.str.contains('|'.join(tarja_keywords), na=False, regex=True) | (cat_lower.str.contains('tarja', na=False))
    es_mueble = desc_lower.str.contains('|'.join(mueble_keywords), na=False, regex=True) | (cat_lower.str.contains('mueble', na=False))
    es_herramienta = desc_lower.str.contains('|'.join(herramienta_keywords), na=False, regex=True) | (cat_lower.str.contains('herramienta', na=False))
    
    # Separar
    df_tinacos = df[es_tinaco].copy()
    df_espejos = df[es_espejo & ~es_tinaco].copy()
    df_tarjas = df[es_tarja & ~es_tinaco & ~es_espejo].copy()
    df_muebles_otras = df[es_mueble & ~es_tinaco & ~es_espejo & ~es_tarja].copy()
    df_herramientas = df[es_herramienta & ~es_tinaco & ~es_espejo & ~es_tarja & ~es_mueble].copy()
    
    # Lo que queda va a otras_restantes.csv
    clasificados = es_tinaco | es_espejo | es_tarja | es_mueble | es_herramienta
    df_otras_restantes = df[~clasificados].copy()
    
    # Guardar
    df_tinacos.to_csv(os.path.join(data_dir, "tinacos.csv"), index=False)
    df_espejos.to_csv(os.path.join(data_dir, "espejos.csv"), index=False)
    df_tarjas.to_csv(os.path.join(data_dir, "tarjas.csv"), index=False)
    df_muebles_otras.to_csv(os.path.join(data_dir, "muebles_otras.csv"), index=False)
    df_herramientas.to_csv(os.path.join(data_dir, "herramientas.csv"), index=False)
    df_otras_restantes.to_csv(os.path.join(data_dir, "otras_restantes.csv"), index=False)
    
    print(f"   ✅ Creados:")
    print(f"      - tinacos.csv: {len(df_tinacos)}")
    print(f"      - espejos.csv: {len(df_espejos)}")
    print(f"      - tarjas.csv: {len(df_tarjas)}")
    print(f"      - muebles_otras.csv: {len(df_muebles_otras)}")
    print(f"      - herramientas.csv: {len(df_herramientas)}")
    print(f"      - otras_restantes.csv: {len(df_otras_restantes)}")
    
    # Renombrar original
    os.rename(otras_path, os.path.join(data_dir, "otras_original.csv"))
else:
    print("   ⏭️  No encontrado")

# 4. UNIFICAR MUEBLES (de griferia y de otras)
print("\n🔨 Unificando muebles...")
muebles_grif_path = os.path.join(data_dir, "muebles_griferia.csv")
muebles_otras_path = os.path.join(data_dir, "muebles_otras.csv")

dfs_muebles = []
if os.path.exists(muebles_grif_path):
    dfs_muebles.append(pd.read_csv(muebles_grif_path))
    os.remove(muebles_grif_path)
if os.path.exists(muebles_otras_path):
    dfs_muebles.append(pd.read_csv(muebles_otras_path))
    os.remove(muebles_otras_path)

if dfs_muebles:
    df_muebles = pd.concat(dfs_muebles, ignore_index=True)
    df_muebles.to_csv(os.path.join(data_dir, "muebles.csv"), index=False)
    print(f"   ✅ muebles.csv: {len(df_muebles)} productos totales")

# 5. LIMPIEZA FINAL
print("\n🧹 Limpiando archivos temporales...")
temporales = ['muebles_griferia.csv', 'muebles_otras.csv']
for temp in temporales:
    temp_path = os.path.join(data_dir, temp)
    if os.path.exists(temp_path):
        os.remove(temp_path)

print("\n📁 Estructura final en data/:")
print("   nacionales.csv          → pisos nacionales")
print("   importados.csv          → pisos importados")
print("   griferia_pura.csv       → grifería (solo llaves, regaderas)")
print("   lavabos.csv             → lavabos")
print("   sanitarios.csv          → wc, tazas, tanques")
print("   muebles.csv             → muebles de baño")
print("   tinacos.csv             → tinacos")
print("   espejos.csv             → espejos")
print("   tarjas.csv              → tarjas y fregaderos")
print("   herramientas.csv        → herramientas")
print("   polvos.csv              → pegamentos, cementos")
print("   otras_restantes.csv     → lo que no clasificó")
print("   promo.csv               → promociones (para metadato)")

print("\n✅ ¡Listo! Backup guardado en {}/".format(backup_dir))