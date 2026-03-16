
# --- CODE LOOKUP GUARD (added) ---
import re
def _is_product_code(q):
    q = (q or "").strip()
    # patrones comunes: letras+digitos, NEOS018, CMX01, PMARBLC, con o sin guiones
    return bool(re.match(r'^[A-Za-z]{2,6}[-_]?\d{1,4}$', q)) or bool(re.match(r'^[A-Za-z]{2,6}\d{1,4}$', q))
def lookup_code_direct(query):
    q = (query or "").strip().lower()
    # 1) try Chroma metadatas fast path
    try:
        import os
        CHROMA_PATH = os.environ.get("CHROMA_PATH","chroma_db_v3")
        if os.path.isdir(CHROMA_PATH):
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            for c in client.list_collections():
                col = client.get_collection(c.name)
                # get small batch of metadatas to search for exact code
                res = col.get(include=['metadatas','documents','ids'], limit=1000)
                metas = res.get('metadatas', [])
                ids = res.get('ids', [])
                docs = res.get('documents', [])
                for i, meta in enumerate(metas):
                    if not meta: continue
                    code = (meta.get('codigo') or meta.get('Codigo') or '').strip().lower()
                    if code == q:
                        precio = meta.get('precio_unitario') or meta.get('precio') or ''
                        desc = meta.get('descripcion') or docs[i] if i < len(docs) else ''
                        return {'codigo': code, 'descripcion': desc, 'precio': precio, 'source': 'chroma'}
    except Exception:
        pass
    # 2) fallback to CSV unified quick scan
    try:
        import csv, os
        csv_path = Path(__file__).parent / 'catalog_work' / 'catalog_unified.csv'
        if csv_path.exists():
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = (row.get('Codigo') or row.get('codigo') or '').strip().lower()
                    if code == q:
                        return {'codigo': code, 'descripcion': row.get('Descripcion') or row.get('RawLine') or '', 'precio': row.get('Precio') or '', 'source': 'csv'}
    except Exception:
        pass
    return None
# --- END CODE LOOKUP GUARD ---


import csv, re
from pathlib import Path

def _normalize(s):
    return re.sub(r'\s+',' ', (s or "").strip().lower())

def find_pisos_from_csv(query, csv_path=Path(__file__).parent / "data" / "importados.csv", max_results=6):
    q = _normalize(query)
    metraje = None
    color = None
    m = re.search(r'(\d+(?:\.\d+)?)\s*m', q)
    if m:
        try:
            metraje = float(m.group(1))
        except:
            metraje = None
    if 'blanco' in q: color = 'blanco'
    if 'gris' in q: color = 'gris'
    results = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                codigo = (row.get('Codigo') or row.get('Codigo ') or '').strip()
                desc = _normalize(row.get('Descripcion ') or row.get('Descripcion') or '')
                coleccion = _normalize(row.get('Proveedor') or row.get('Coleccion') or '')
                score = 0
                if 'neos' in codigo.lower(): score += 2
                if 'piso' in desc or 'piso' in coleccion: score += 1
                if color and color in desc: score += 2
                if metraje and ('120' in (row.get('Formato') or '') or '60' in (row.get('Formato') or '')): score += 1
                if score>0:
                    results.append({
                        "codigo": codigo,
                        "descripcion": row.get('Descripcion ') or row.get('Descripcion') or '',
                        "precio": (row.get('Precio sistema Caja') or row.get('Precio M2 con complementos') or '').strip(),
                        "score": score
                    })
        results.sort(key=lambda x: (-x['score'], x['codigo']))
        return results[:max_results]
    except Exception:
        return []
# --- CHROMA FALLBACK (added by patch) ---
import os
def chroma_search(query, top_k=6):
    """
    Intentará usar chromadb PersistentClient si está disponible en CHROMA_PATH.
    Devuelve lista de dicts {codigo, descripcion, precio, score} o [].
    No falla si chroma no está instalado o no existe la DB.
    """
    try:
        CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db_v3")
        if not os.path.isdir(CHROMA_PATH):
            return []
        import chromadb
        from chromadb.utils import embedding_functions
        # cliente persistente (no lanza si path inválido)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        # suponer colección por nombre 'catalog' o por cada colección; intentar 'catalog' primero
        col = None
        for name in client.list_collections():
            if 'catalog' in name.lower():
                col = client.get_collection(name)
                break
        if col is None:
            # fallback: tomar la primera colección disponible
            cols = client.list_collections()
            if not cols:
                return []
            col = client.get_collection(cols[0])
        # consulta semántica
        res = col.query(query_texts=[query], n_results=top_k)
        out = []
        # res['metadatas'] y res['ids'] y res['distances'] pueden variar según versión
        metadatas = res.get('metadatas', [])
        ids = res.get('ids', [])
        distances = res.get('distances', [])
        for i, meta in enumerate(metadatas[0] if metadatas else []):
            codigo = meta.get('codigo') or meta.get('Codigo') or ids[0][i] if ids else f"ID_{i}"
            descripcion = meta.get('descripcion') or meta.get('Descripcion') or str(meta)
            precio = meta.get('precio') or meta.get('Precio') or ''
            score = 1.0
            # si chroma devuelve distancia, convertir a score aproximado
            try:
                d = distances[0][i] if distances else None
                if d is not None:
                    score = max(0.0, 1.0 - float(d))
            except:
                pass
            out.append({"codigo": codigo, "descripcion": descripcion, "precio": precio, "score": score})
        return out
    except Exception as e:
        # no romper el flujo si chroma no está disponible
        return []
# --- END CHROMA FALLBACK ---
import os, time, logging


# --- ENHANCED CODE LOOKUP + INDEX (added) ---
import re, os, json
from pathlib import Path

_CODES_INDEX_PATH = Path(__file__).parent / "catalog_work" / "codes_index.json"

def _normalize_code(q):
    if not q: return ""
    q = str(q).strip().lower()
    q = re.sub(r'[\s\-_]+', '', q)
    return q

def _is_product_code(q):
    qn = _normalize_code(q)
    return bool(re.match(r'^[a-z]{1,8}\d{1,8}$', qn))

def _build_codes_index(force=False):
    """
    Construye un índice local JSON con mapeo code -> {source, collection, descripcion, precio}
    Lee Chroma (si existe) y CSV unificado. No modifica Chroma.
    """
    try:
        if _CODES_INDEX_PATH.exists() and not force:
            return True
        idx = {}
        # 1) Chroma: recorrer colecciones y metadatas/documents
        try:
            CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db_v3")
            if os.path.isdir(CHROMA_PATH):
                import chromadb
                client = chromadb.PersistentClient(path=CHROMA_PATH)
                for c in client.list_collections():
                    try:
                        col = client.get_collection(c.name)
                        # obtener bloques razonables; si la colección es grande, iterar en chunks
                        res = col.get(include=['metadatas','documents','ids'], limit=5000)
                        metas = res.get('metadatas', []) or []
                        docs = res.get('documents', []) or []
                        for i, meta in enumerate(metas):
                            if not isinstance(meta, dict):
                                continue
                            # posibles claves de código
                            for key in ('codigo','Codigo','code','sku','SKU','item_code'):
                                code_raw = meta.get(key)
                                if code_raw:
                                    code = _normalize_code(code_raw)
                                    if code and code not in idx:
                                        descripcion = meta.get('descripcion') or (docs[i] if i < len(docs) else '')
                                        precio = meta.get('precio_unitario') or meta.get('precio') or ''
                                        idx[code] = {'codigo': (meta.get(key) or '').upper(), 'descripcion': descripcion, 'precio': precio, 'source': f'chroma:{c.name}'}
                        # también buscar en documents si no hay metadatas útiles
                        for i, doc in enumerate(docs):
                            if not doc: continue
                            # buscar tokens tipo CMX01 dentro del documento
                            for token in re.findall(r'[A-Za-z]{2,6}[-_]?\\d{1,6}', doc):
                                code = _normalize_code(token)
                                if code and code not in idx:
                                    idx[code] = {'codigo': token.upper(), 'descripcion': doc[:120], 'precio': '', 'source': f'chroma_doc:{c.name}'}
                    except Exception:
                        continue
        except Exception:
            pass

        # 2) CSV unified fallback
        try:
            csv_path = Path(__file__).parent / 'catalog_work' / 'catalog_unified.csv'
            if csv_path.exists():
                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # posibles nombres de columna
                        raw_code = (row.get('Codigo') or row.get('codigo') or row.get('SKU') or row.get('sku') or row.get('code') or row.get('Code') or '')
                        code = _normalize_code(raw_code)
                        if not code: continue
                        if code not in idx:
                            descripcion = row.get('Descripcion') or row.get('Descripcion ') or row.get('RawLine') or ''
                            precio = row.get('Precio') or row.get('Precio sistema Caja') or row.get('Precio M2 con complementos') or ''
                            idx[code] = {'codigo': (raw_code or '').upper(), 'descripcion': descripcion, 'precio': precio, 'source': 'csv'}
        except Exception:
            pass

        # guardar índice
        _CODES_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CODES_INDEX_PATH, 'w', encoding='utf-8') as fh:
            json.dump(idx, fh, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def lookup_code_direct(query):
    """
    Lookup rápido usando índice local si existe; si no existe, intenta construirlo.
    Devuelve dict {'codigo','descripcion','precio','source'} o None.
    """
    try:
        qnorm = _normalize_code(query)
        if not qnorm:
            return None
        # asegurar índice
        if not _CODES_INDEX_PATH.exists():
            _build_codes_index()
        if _CODES_INDEX_PATH.exists():
            with open(_CODES_INDEX_PATH, 'r', encoding='utf-8') as fh:
                idx = json.load(fh)
            if qnorm in idx:
                return idx[qnorm]
        # fallback: intentar búsqueda directa en Chroma/CSV (por si el índice no se pudo construir)
        # (mantener compatibilidad con implementaciones previas)
        # Intentar Chroma directo (sin index)
        try:
            CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db_v3")
            if os.path.isdir(CHROMA_PATH):
                import chromadb
                client = chromadb.PersistentClient(path=CHROMA_PATH)
                for c in client.list_collections():
                    try:
                        col = client.get_collection(c.name)
                        res = col.get(include=['metadatas','documents','ids'], limit=5000)
                        metas = res.get('metadatas', []) or []
                        docs = res.get('documents', []) or []
                        for i, meta in enumerate(metas):
                            if not isinstance(meta, dict):
                                continue
                            for key in ('codigo','Codigo','code','sku','SKU','item_code'):
                                code_raw = meta.get(key)
                                if code_raw and _normalize_code(code_raw) == qnorm:
                                    precio = meta.get('precio_unitario') or meta.get('precio') or ''
                                    descripcion = meta.get('descripcion') or (docs[i] if i < len(docs) else '')
                                    return {'codigo': (meta.get(key) or '').upper(), 'descripcion': descripcion, 'precio': precio, 'source': f'chroma:{c.name}'}
                        # buscar en documentos
                        for doc in docs:
                            if not doc: continue
                            if qnorm in _normalize_code(doc):
                                return {'codigo': query.upper(), 'descripcion': doc[:120], 'precio': '', 'source': f'chroma_doc:{c.name}'}
                    except Exception:
                        continue
        except Exception:
            pass
        # CSV fallback
        try:
            csv_path = Path(__file__).parent / 'catalog_work' / 'catalog_unified.csv'
            if csv_path.exists():
                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        raw_code = (row.get('Codigo') or row.get('codigo') or row.get('SKU') or row.get('sku') or row.get('code') or '')
                        if _normalize_code(raw_code) == qnorm:
                            descripcion = row.get('Descripcion') or row.get('RawLine') or ''
                            precio = row.get('Precio') or ''
                            return {'codigo': (raw_code or '').upper(), 'descripcion': descripcion, 'precio': precio, 'source': 'csv'}
        except Exception:
            pass
        return None
    except Exception:
        return None
# --- END ENHANCED CODE LOOKUP + INDEX ---

logger = logging.getLogger(__name__)

# Confidence threshold under which we fallback to chroma
CHROMA_CONF_THRESHOLD = float(os.environ.get("CHROMA_CONF_THRESHOLD", "0.45"))

def _format_result(codigo, descripcion, precio, source, updated_at=None, confidence=1.0):
    return {
        "codigo": codigo,
        "descripcion": descripcion,
        "precio": precio,
        "source": source,
        "updated_at": updated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confidence": float(confidence)
    }

def hybrid_search(query, max_results=6):
    """
    1) Exact search in local collections/CSV/DB (fast, authoritative)
    2) If no results or low confidence, fallback to chroma_search (semantic)
    Returns list of formatted results.
    """
    # 1) Exact search: try code match first
    try:
        exact = []
        q = query.strip().lower()
        # if query looks like a code (alphanumeric short), try direct lookup
        if len(q) <= 12 and any(ch.isdigit() for ch in q):
            # try to find by code in Chroma metadatas or CSV quick scan
            # prefer collection-level exact lookup if available
            try:
                import chromadb
                CHROMA_PATH = os.environ.get("CHROMA_PATH","chroma_db_v3")
                if os.path.isdir(CHROMA_PATH):
                    client = chromadb.PersistentClient(path=CHROMA_PATH)
                    # search each collection metadata for exact code
                    for c in client.list_collections():
                        col = client.get_collection(c.name)
                        # query by metadata filter if supported
                        try:
                            res = col.query(query_texts=[query], n_results=1, where={"codigo": q})
                            metas = res.get("metadatas", [[]])[0]
                            ids = res.get("ids", [[]])[0]
                            docs = res.get("documents", [[]])[0]
                            if metas:
                                for i, m in enumerate(metas):
                                    exact.append(_format_result(m.get("codigo", ids[i] if i < len(ids) else ""), m.get("descripcion", docs[i] if i < len(docs) else ""), m.get("precio_unitario", m.get("precio","")), "chroma", m.get("updated_at",""), 1.0))
                        except Exception:
                            # fallback to scanning documents for code substring
                            try:
                                getres = col.get(include=['metadatas','documents','ids'], limit=1000)
                                for i, meta in enumerate(getres.get('metadatas',[])):
                                    if meta and meta.get('codigo','').lower() == q:
                                        exact.append(_format_result(meta.get('codigo'), meta.get('descripcion',''), meta.get('precio_unitario',''), "chroma", meta.get('updated_at',''), 1.0))
                            except Exception:
                                pass
            except Exception:
                pass

        # 2) Token match scan in CSV fallback (fast, best-effort)
        if not exact:
            csv_path = os.path.join(os.path.dirname(__file__), "catalog_work", "catalog_unified.csv")
            if os.path.isfile(csv_path):
                import csv
                qtokens = set([t for t in q.split() if len(t)>1])
                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        desc = (row.get('Descripcion') or row.get('RawLine') or "").lower()
                        score = sum(1 for t in qtokens if t in desc)
                        if score>0:
                            exact.append(_format_result(row.get('Codigo',''), row.get('Descripcion',''), row.get('Precio',''), "csv", row.get('RawLine',''), float(score)/max(1,len(qtokens))))
                        if len(exact) >= max_results:
                            break

        # sort exact by confidence
        exact = sorted(exact, key=lambda x: -x['confidence'])
        if exact:
            # if top confidence is high, return exact results
            if exact[0]['confidence'] >= 0.6:
                return exact[:max_results]
    except Exception as e:
        logger.exception("hybrid exact search error: %s", e)

    # 3) Fallback to chroma semantic search
    try:
        chroma_matches = chroma_search(query, top_k=max_results)
        # normalize chroma confidence to 0..1 and filter by threshold
        filtered = []
        for m in chroma_matches:
            conf = float(m.get('score', 0.0))
            if conf >= CHROMA_CONF_THRESHOLD:
                filtered.append(_format_result(m.get('codigo', ''), m.get('descripcion',''), m.get('precio',''), "chroma", m.get('updated_at',''), conf))
        if filtered:
            return filtered[:max_results]
    except Exception as e:
        logger.exception("chroma fallback error: %s", e)

    # 4) final fallback: return exact even if low confidence or empty list
    return exact[:max_results] if exact else []
