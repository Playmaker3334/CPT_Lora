import fnmatch
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

TESTADO = re.compile(r"\*(?:[ \t]*\*)+")
PARRAFO = re.compile(r"\n[ \t]*\n")
ESPACIOS = re.compile(r"\s+")

TRAZABILIDAD = ("fuente", "id_documento", "version_pipeline_limpieza", "sha256_origen", "origen_texto", "particion")


def leer_manifest(ruta):
    if not ruta:
        return None
    with open(ruta, encoding="utf-8") as f:
        return {(d["fuente"], str(d["id_documento"])) for d in (json.loads(l) for l in f if l.strip())}


def coincide(fuente, patrones):
    return any(fnmatch.fnmatch(fuente, p) for p in patrones or ())


def particion(clave, fraccion_val, siempre_train=()):
    if coincide(clave[0], siempre_train):
        return "train"
    h = int(hashlib.sha256(f"{clave[0]}\x1f{clave[1]}".encode()).hexdigest()[:8], 16)
    return "val" if h / 0xFFFFFFFF < fraccion_val else "train"


def valor(doc, campo):
    if "." not in campo:
        return (doc.get("metadata") or {}).get(campo)
    actual = doc
    for parte in campo.split("."):
        if not isinstance(actual, dict):
            return None
        actual = actual.get(parte)
    return actual


def numero(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pasa_filtros(doc, filtros, exento=False):
    meta = doc.get("metadata") or {}
    if filtros.get("versiones") and doc.get("version_pipeline_limpieza") not in filtros["versiones"]:
        return False
    if filtros.get("fuentes") and not coincide(doc["fuente"], filtros["fuentes"]):
        return False
    if filtros.get("materias") and meta.get("materia") not in filtros["materias"]:
        return False
    if filtros.get("origen_texto") and meta.get("origen_texto") not in filtros["origen_texto"]:
        return False
    if exento:
        return True
    for campo, permitidos in (filtros.get("campos") or {}).items():
        if permitidos and valor(doc, campo) not in permitidos:
            return False
    for campo, (minimo, maximo) in (filtros.get("rangos") or {}).items():
        minimo, maximo = numero(minimo), numero(maximo)
        if minimo is None and maximo is None:
            continue
        v = numero(valor(doc, campo))
        if v is None or (minimo is not None and v < minimo) or (maximo is not None and v > maximo):
            return False
    return True


def encabezado(doc, campos, nulos, titulos):
    partes = []
    for campo in campos:
        v = titulos.get(str(doc["id_documento"])) if campo == "titulo" else valor(doc, campo)
        if v is None:
            if nulos == "omitir":
                continue
            v = nulos
        partes.append(f"{campo.rsplit('.', 1)[-1]}: {v}")
    return " | ".join(partes) + "\n" if partes else ""


def spans_testado(texto):
    return [[m.start(), m.end()] for m in TESTADO.finditer(texto)]


def parrafos(texto):
    ini = 0
    for m in PARRAFO.finditer(texto):
        yield ini, m.start()
        ini = m.end()
    yield ini, len(texto)


def firma_parrafo(texto, a, b, min_caracteres):
    normal = ESPACIOS.sub(" ", TESTADO.sub("", texto[a:b])).strip()
    return hashlib.sha1(normal.encode()).hexdigest() if len(normal) >= min_caracteres else None


def spans_duplicados(docs, max_repeticiones, min_caracteres):
    conteo = Counter()
    for d in docs:
        for a, b in parrafos(d["texto"]):
            f = firma_parrafo(d["texto"], a, b, min_caracteres)
            if f:
                conteo[f] += 1
    vistos = Counter()
    for d in docs:
        spans = []
        for a, b in parrafos(d["texto"]):
            f = firma_parrafo(d["texto"], a, b, min_caracteres)
            if f and conteo[f] > max_repeticiones:
                vistos[f] += 1
                if vistos[f] > max_repeticiones:
                    spans.append([a, b])
        d["spans"] = d["spans"] + spans
    return sum(1 for c in conteo.values() if c > max_repeticiones)


def tope(fuente, maximos):
    if not maximos:
        return None
    if isinstance(maximos, int):
        return maximos
    return next((n for patron, n in maximos.items() if fnmatch.fnmatch(fuente, patron)), None)


def orden_estable(ruta, raiz):
    return hashlib.sha256(ruta.relative_to(raiz).as_posix().encode()).hexdigest()


def leer_juridico(cfg):
    raiz = Path(cfg["raiz"])
    incluidos = leer_manifest(cfg.get("manifest"))
    enc = cfg["encabezado"]
    titulos = cfg.get("titulos") or {}
    maximos = cfg.get("max_documentos")
    docs = []
    por_fuente = Counter()
    for ruta in sorted(raiz.rglob("*.json"), key=lambda r: orden_estable(r, raiz)):
        with open(ruta, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or "texto" not in doc:
            continue
        clave = (doc["fuente"], str(doc["id_documento"]))
        if incluidos is not None and clave not in incluidos:
            continue
        if not pasa_filtros(doc, cfg["filtros"], coincide(doc["fuente"], cfg.get("siempre_train"))):
            continue
        limite = tope(doc["fuente"], maximos)
        if limite is not None and por_fuente[doc["fuente"]] >= limite:
            continue
        por_fuente[doc["fuente"]] += 1
        meta = doc.get("metadata") or {}
        docs.append(
            {
                "texto": doc["texto"],
                "encabezado": encabezado(doc, enc["campos"], enc["nulos"], titulos),
                "spans": spans_testado(doc["texto"]) if cfg["testado"] == "enmascarar" else [],
                "fuente": doc["fuente"],
                "id_documento": clave[1],
                "version_pipeline_limpieza": doc.get("version_pipeline_limpieza") or "",
                "sha256_origen": doc.get("sha256_origen") or "",
                "origen_texto": meta.get("origen_texto") or "",
                "particion": particion(clave, cfg["fraccion_val"], cfg.get("siempre_train")),
            }
        )
    sin_titulo = sorted(
        d["id_documento"] for d in docs if coincide(d["fuente"], cfg.get("siempre_train")) and d["id_documento"] not in titulos
    )
    if "titulo" in enc["campos"] and sin_titulo:
        print(f"aviso: documentos sin titulo en el encabezado: {sin_titulo}")
    return docs


def leer_repaso(cfg):
    rep = cfg.get("repaso") or {}
    if not rep.get("ruta"):
        return []
    docs = []
    with open(rep["ruta"], encoding="utf-8") as f:
        for linea in (l for l in f if l.strip()):
            texto = json.loads(linea)[rep["campo"]]
            huella = hashlib.sha256(texto.encode()).hexdigest()
            clave = ("repaso", huella[:16])
            docs.append(
                {
                    "texto": texto,
                    "encabezado": "",
                    "spans": [],
                    "fuente": "repaso",
                    "id_documento": clave[1],
                    "version_pipeline_limpieza": "externo",
                    "sha256_origen": huella,
                    "origen_texto": "nativo",
                    "particion": particion(clave, cfg["fraccion_val"]),
                }
            )
            if rep.get("max_documentos") and len(docs) >= rep["max_documentos"]:
                break
    return docs


def documentos(cfg):
    juridico = leer_juridico(cfg)
    dup = cfg["duplicados"]
    if dup["politica"] == "enmascarar":
        n = spans_duplicados(juridico, dup["max_repeticiones"], dup["min_caracteres"])
        print(f"parrafos repetidos por encima del umbral: {n}")
    repaso = leer_repaso(cfg)
    conteo = Counter((d["fuente"], d["particion"]) for d in juridico + repaso)
    for (fuente, part), n in sorted(conteo.items()):
        print(f"{fuente:<24} {part:<5} {n}")
    return juridico + repaso
