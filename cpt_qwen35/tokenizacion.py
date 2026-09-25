import bisect
import re

import numpy as np
from datasets import Dataset, Features, Sequence, Value

from .corpus import TRAZABILIDAD

IGNORAR = -100
PARRAFO = re.compile(r"\n[ \t]*\n")
ORACION = re.compile(r"(?<=[.;:])\s+(?=[A-ZÁÉÍÓÚÑ0-9¿¡\"«(])")

ENTRADA = Features(
    {
        "texto": Value("string"),
        "encabezado": Value("string"),
        "spans": Sequence(Sequence(Value("int64"))),
        **{c: Value("string") for c in TRAZABILIDAD},
    }
)


def fronteras(texto, inicios, patron):
    return sorted({bisect.bisect_left(inicios, m.end()) for m in patron.finditer(texto)})


def trocear(texto, offsets, max_tokens, fraccion_minima=0.5):
    n = len(offsets)
    inicios = [s for s, _ in offsets]
    niveles = (fronteras(texto, inicios, PARRAFO), fronteras(texto, inicios, ORACION))
    cortes = []
    ini = 0
    while n - ini > max_tokens:
        tope = ini + max_tokens
        piso = ini + int(max_tokens * fraccion_minima)
        fin = tope
        for f in niveles:
            j = bisect.bisect_right(f, tope) - 1
            if j >= 0 and f[j] > piso:
                fin = f[j]
                break
        cortes.append((ini, fin))
        ini = fin
    if ini < n:
        cortes.append((ini, n))
    return cortes


def fusionar_spans(spans):
    salida = []
    for a, b in sorted((int(a), int(b)) for a, b in spans if b > a):
        if salida and a <= salida[-1][1]:
            salida[-1][1] = max(salida[-1][1], b)
        else:
            salida.append([a, b])
    return salida


def mascara_tokens(offsets, spans):
    mascara = np.zeros(len(offsets), dtype=bool)
    spans = fusionar_spans(spans or [])
    if not spans or not len(offsets):
        return mascara
    o = np.asarray(offsets, dtype=np.int64)
    s = np.asarray(spans, dtype=np.int64)
    j = np.searchsorted(s[:, 0], o[:, 1], side="left") - 1
    valido = (j >= 0) & (o[:, 1] > o[:, 0])
    mascara[valido] = s[j[valido], 1] > o[valido, 0]
    return mascara


class Segmentador:
    def __init__(self, tokenizador, max_len, perdida_encabezado):
        if tokenizador.eos_token_id is None:
            raise ValueError("El tokenizador no define eos_token_id.")
        self.tok = tokenizador
        self.max_len = max_len
        self.perdida_encabezado = perdida_encabezado
        self.eos = tokenizador.eos_token_id

    def __call__(self, lote):
        salida = {k: [] for k in ("input_ids", "labels", "n", "n_enmascarados", "indice_segmento", *TRAZABILIDAD)}
        cuerpos = self.tok(lote["texto"], add_special_tokens=False, return_offsets_mapping=True)
        encabezados = self.tok(lote["encabezado"], add_special_tokens=False)["input_ids"]
        for i, texto in enumerate(lote["texto"]):
            ids = cuerpos["input_ids"][i]
            offsets = cuerpos["offset_mapping"][i]
            enc = encabezados[i]
            presupuesto = self.max_len - len(enc) - 1
            if presupuesto < 1:
                raise ValueError(f"Encabezado de {len(enc)} tokens no cabe en max_len={self.max_len}.")
            enmascarado = mascara_tokens(offsets, lote["spans"][i])
            etiqueta_enc = list(enc) if self.perdida_encabezado else [IGNORAR] * len(enc)
            cortes = trocear(texto, offsets, presupuesto)
            for k, (a, b) in enumerate(cortes):
                x = list(enc) + ids[a:b]
                y = etiqueta_enc + [IGNORAR if enmascarado[t] else ids[t] for t in range(a, b)]
                if k == len(cortes) - 1:
                    x.append(self.eos)
                    y.append(self.eos)
                if all(v == IGNORAR for v in y[1:]):
                    continue
                salida["input_ids"].append(x)
                salida["labels"].append(y)
                salida["n"].append(len(x))
                salida["n_enmascarados"].append(int(enmascarado[a:b].sum()))
                salida["indice_segmento"].append(k)
                for campo in TRAZABILIDAD:
                    salida[campo].append(lote[campo][i])
        return salida


def construir(docs, tokenizador, cfg, destino):
    base = Dataset.from_list(docs, features=ENTRADA)
    segmentos = base.map(
        Segmentador(tokenizador, cfg["seq_len"], cfg["encabezado"]["perdida"]),
        batched=True,
        batch_size=64,
        remove_columns=base.column_names,
        num_proc=cfg["num_proc"],
    )
    for part in ("train", "val"):
        sub = segmentos.filter(lambda p: p == part, input_columns="particion", num_proc=cfg["num_proc"])
        sub.save_to_disk(str(destino / part / "segmentos"))
        tokens = sum(sub["n"])
        enmascarados = sum(sub["n_enmascarados"])
        print(
            f"{part}: segmentos={len(sub)} tokens={tokens} "
            f"enmascarados={enmascarados} ({enmascarados / max(tokens, 1):.4f})"
        )
