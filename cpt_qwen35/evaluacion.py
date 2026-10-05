import contextlib
import difflib
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import PeftModel

from .tokenizacion import IGNORAR

LOTES = {"juridico": lambda fuente: fuente != "repaso", "general": lambda fuente: fuente == "repaso"}


def _hash(items):
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def _claves(segmentos):
    columnas = segmentos.select_columns(["fuente", "id_documento"])
    return set(zip(columnas["fuente"], (str(x) for x in columnas["id_documento"])))


def construir_lotes(cfg, dir_datos, semilla):
    destino = Path(cfg["lotes"])
    destino.mkdir(parents=True, exist_ok=True)
    val = load_from_disk(str(Path(dir_datos) / "val" / "segmentos"))
    claves_train = _claves(load_from_disk(str(Path(dir_datos) / "train" / "segmentos")))
    fuentes = val["fuente"]
    rng = np.random.default_rng(semilla)
    for nombre, condicion in LOTES.items():
        ruta = destino / f"{nombre}.json"
        if ruta.exists():
            items = json.loads(ruta.read_text())["items"]
            contaminados = {(it["fuente"], str(it["id_documento"])) for it in items} & claves_train
            if contaminados:
                raise SystemExit(
                    f"Lote {nombre}: {len(contaminados)} documentos están en train (¿cambió fraccion_val o la clave "
                    f"del repaso?). Borra el lote para regenerarlo; dejará de ser comparable con corridas anteriores."
                )
            print(f"lote {nombre}: ya existe, no se regenera; sin documentos en train")
            continue
        candidatos = [i for i, f in enumerate(fuentes) if condicion(f)]
        if not candidatos:
            print(f"lote {nombre}: sin segmentos de validación")
            continue
        elegidos = sorted(rng.choice(candidatos, min(cfg["n_segmentos"], len(candidatos)), replace=False).tolist())
        items = [
            {
                "input_ids": r["input_ids"][: cfg["max_tokens"]],
                "labels": r["labels"][: cfg["max_tokens"]],
                "fuente": r["fuente"],
                "origen_texto": r["origen_texto"],
                "id_documento": r["id_documento"],
            }
            for r in val.select(elegidos)
        ]
        ruta.write_text(json.dumps({"sha256": _hash(items), "items": items}))
        print(f"lote {nombre}: {len(items)} segmentos")


def cargar_lotes(cfg):
    lotes = {}
    for ruta in sorted(Path(cfg["lotes"]).glob("*.json")):
        datos = json.loads(ruta.read_text())
        if _hash(datos["items"]) != datos["sha256"]:
            raise SystemExit(f"El lote {ruta} fue modificado: el hash no coincide.")
        lotes[ruta.stem] = datos["items"]
    return lotes


def _nucleo(modelo):
    base = modelo.get_base_model() if isinstance(modelo, PeftModel) else modelo
    return base.model, base.lm_head


def _variantes(modelo):
    yield "ajustado", contextlib.nullcontext()
    if isinstance(modelo, PeftModel):
        yield "base", modelo.disable_adapter()


def _ocultos(backbone, ids):
    return backbone(input_ids=ids, use_cache=False).last_hidden_state[0]


def _metricas(v):
    t = max(v[0], 1.0)
    loss, loss_base = v[1] / t, v[4] / t
    return {
        "tokens": int(v[0]),
        "loss": loss,
        "perplejidad": math.exp(min(loss, 50)),
        "entropia": v[2] / t,
        "kl_contra_base": v[3] / t,
        "loss_base": loss_base,
        "ganancia": loss_base - loss,
    }


@torch.no_grad()
def sondear(modelo, lotes, fragmento=512):
    estaba = modelo.training
    modelo.eval()
    backbone, cabeza = _nucleo(modelo)
    comparar = isinstance(modelo, PeftModel)
    dispositivo = cabeza.weight.device
    resultado = {}
    for nombre, items in lotes.items():
        acumulado = defaultdict(lambda: np.zeros(5))
        for it in items:
            ids = torch.tensor([it["input_ids"]], device=dispositivo)
            objetivo = torch.tensor(it["labels"][1:], device=dispositivo)
            h = _ocultos(backbone, ids)[:-1]
            if comparar:
                with modelo.disable_adapter():
                    hb = _ocultos(backbone, ids)[:-1]
            suma = np.zeros(5)
            for a in range(0, len(objetivo), fragmento):
                v = objetivo[a : a + fragmento] != IGNORAR
                if not v.any():
                    continue
                t = objetivo[a : a + fragmento][v]
                lp = F.log_softmax(cabeza(h[a : a + fragmento][v]).float(), dim=-1)
                nll = -lp.gather(1, t[:, None]).squeeze(1)
                p = lp.exp()
                ent = -(p * lp).sum(-1)
                if comparar:
                    lq = F.log_softmax(cabeza(hb[a : a + fragmento][v]).float(), dim=-1)
                    kl = (p * (lp - lq)).sum(-1)
                    nll_base = -lq.gather(1, t[:, None]).squeeze(1)
                else:
                    kl = torch.zeros_like(nll)
                    nll_base = nll
                suma += [v.sum().item(), nll.sum().item(), ent.sum().item(), kl.sum().item(), nll_base.sum().item()]
            for clave in ("total", f"fuente={it['fuente']}", f"origen_texto={it['origen_texto']}"):
                acumulado[clave] += suma
        resultado[nombre] = {k: _metricas(v) for k, v in sorted(acumulado.items())}
    modelo.train(estaba)
    return resultado


@torch.no_grad()
def _logverosimilitud(backbone, cabeza, contexto, candidato):
    ids = torch.tensor([contexto + candidato], device=cabeza.weight.device)
    h = _ocultos(backbone, ids)[len(contexto) - 1 : -1]
    lp = F.log_softmax(cabeza(h).float(), dim=-1)
    return lp.gather(1, torch.tensor(candidato, device=lp.device)[:, None]).sum().item()


def sondas(modelo, tok, ruta, normalizar):
    backbone, cabeza = _nucleo(modelo)
    conteo = defaultdict(lambda: defaultdict(int))
    with open(ruta, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    for it in items:
        contexto = tok(it["contexto"], add_special_tokens=False)["input_ids"]
        candidatos = [tok(c, add_special_tokens=False)["input_ids"] for c in it["candidatos"]]
        conteo[it["sonda"]]["n"] += 1
        for etiqueta, ctx in _variantes(modelo):
            with ctx:
                puntos = [_logverosimilitud(backbone, cabeza, contexto, c) for c in candidatos]
            if normalizar:
                puntos = [p / len(c) for p, c in zip(puntos, candidatos)]
            conteo[it["sonda"]][etiqueta] += int(int(np.argmax(puntos)) == it["correcta"])
    return {s: {k: (v / c["n"] if k != "n" else v) for k, v in c.items()} for s, c in sorted(conteo.items())}


def generar(modelo, tok, ruta, max_nuevos):
    with open(ruta, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    dispositivo = _nucleo(modelo)[1].weight.device
    salida = []
    for it in items:
        ids = torch.tensor([tok(it["prefijo"], add_special_tokens=False)["input_ids"]], device=dispositivo)
        registro = {"id": it.get("id"), "prefijo": it["prefijo"]}
        referencia = tok(it["referencia"], add_special_tokens=False)["input_ids"] if it.get("referencia") else None
        for etiqueta, ctx in _variantes(modelo):
            with ctx, torch.no_grad():
                out = modelo.generate(
                    input_ids=ids,
                    max_new_tokens=max_nuevos,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
            nuevos = out[0, ids.shape[1] :].tolist()
            r = {"texto": tok.decode(nuevos, skip_special_tokens=True)}
            if referencia:
                ref = referencia[: len(nuevos)]
                comun = next((i for i, (a, b) in enumerate(zip(nuevos, ref)) if a != b), min(len(nuevos), len(ref)))
                r["prefijo_literal_tokens"] = comun
                r["similitud_literal"] = difflib.SequenceMatcher(a=nuevos, b=ref, autojunk=False).ratio()
            registro[etiqueta] = r
        salida.append(registro)
    return salida


def evaluar(modelo, tok, cfg, destino):
    resultado = {"sondeo": sondear(modelo, cargar_lotes(cfg), cfg["fragmento"])}
    if cfg.get("sondas"):
        resultado["sondas"] = sondas(modelo, tok, cfg["sondas"], cfg["normalizar_sondas"])
    if cfg.get("prefijos"):
        resultado["generacion"] = generar(modelo, tok, cfg["prefijos"], cfg["max_nuevos_tokens"])
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    Path(destino).write_text(json.dumps(resultado, ensure_ascii=False, indent=2))
    return resultado
