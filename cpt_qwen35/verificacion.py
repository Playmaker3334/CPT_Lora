from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

from .empaquetado import Colador
from .modelo import MODULOS_ATN, MODULOS_GDN, MODULOS_MLP, aplicar_lora, cargar, exigir_kernels


def _a_dispositivo(lote, dispositivo):
    return {k: (v.to(dispositivo) if torch.is_tensor(v) else v) for k, v in lote.items()}


@torch.no_grad()
def _logits(modelo, lote):
    lote = {k: v for k, v in lote.items() if k != "labels"}
    return modelo(**_a_dispositivo(lote, modelo.device)).logits[0].float()


def verificar(cfg, n_tokens=1024):
    exigir_kernels()
    seg = load_from_disk(str(Path(cfg["datos"]["salida"]) / "train" / "segmentos"))
    candidatos = np.flatnonzero(np.asarray(seg["n"]) >= n_tokens)[:2]
    if len(candidatos) < 2:
        raise SystemExit(f"No hay dos segmentos de al menos {n_tokens} tokens.")
    A, B = (seg[int(i)] for i in candidatos)
    A = {"input_ids": A["input_ids"][:n_tokens], "labels": A["labels"][:n_tokens]}
    B = {"input_ids": B["input_ids"][:n_tokens], "labels": B["labels"][:n_tokens]}

    modelo = cargar(cfg["modelo"], liger=False).eval()
    aplanar = Colador().aplanar
    solo = _logits(modelo, aplanar([B]))
    lote = aplanar([A, B])
    junto = _logits(modelo, lote)[n_tokens:]
    control = _logits(modelo, {k: lote[k] for k in ("input_ids", "position_ids")})[n_tokens:]
    d_ok = (solo - junto).abs().max().item()
    d_control = (solo - control).abs().max().item()
    acuerdo = (solo.argmax(-1) == junto.argmax(-1)).float().mean().item()
    print(f"empaquetado vs aislado: max|dif|={d_ok:.5f} acuerdo_argmax={acuerdo:.4f}")
    print(f"control sin cu_seq_lens ni seq_idx: max|dif|={d_control:.5f}")
    print("aislamiento: OK" if d_ok < 0.1 * d_control else "aislamiento: FALLA")
    del modelo
    torch.cuda.empty_cache()

    modelo = aplicar_lora(cargar(cfg["modelo"], liger=True), cfg["lora"])
    modelo.train()
    salida = modelo(**_a_dispositivo(Colador()([[(A["input_ids"], A["labels"]), (B["input_ids"], B["labels"])]]), modelo.device))
    salida.loss.backward()
    print(f"loss={salida.loss.item():.4f} finita={torch.isfinite(salida.loss).item()}")
    normas = {}
    for nombre, p in modelo.named_parameters():
        if "lora_B" in nombre and p.grad is not None:
            clave = nombre.split(".lora_B")[0].rsplit(".", 1)[-1]
            normas[clave] = normas.get(clave, 0.0) + p.grad.float().norm().item() ** 2
    for clave in MODULOS_GDN + MODULOS_ATN + MODULOS_MLP:
        print(f"grad lora_B {clave}: {normas.get(clave, 0.0) ** 0.5:.3e}")
