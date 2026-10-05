import fnmatch
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorWithFlattening


def bfd(longitudes, max_len):
    libres = SortedList()
    contenedores = []
    for i in np.argsort(-longitudes, kind="stable"):
        n = int(longitudes[i])
        j = libres.bisect_left((n, -1))
        if j < len(libres):
            capacidad, c = libres.pop(j)
            contenedores[c].append(int(i))
            if capacidad > n:
                libres.add((capacidad - n, c))
        else:
            contenedores.append([int(i)])
            if max_len > n:
                libres.add((max_len - n, len(contenedores) - 1))
    return contenedores


def factor(fuente, repeticiones):
    for patron, f in repeticiones.items():
        if fnmatch.fnmatch(fuente, patron):
            return float(f)
    return 1.0


def copias(fuentes, repeticiones, rng):
    f = np.array([factor(x, repeticiones or {}) for x in fuentes])
    return np.floor(f).astype(np.int64) + (rng.random(len(f)) < f - np.floor(f))


def empaquetar(dir_particion, max_len, repeticiones, semilla):
    dir_particion = Path(dir_particion)
    seg = load_from_disk(str(dir_particion / "segmentos")).select_columns(["n", "fuente"])
    n = np.asarray(seg["n"], dtype=np.int64)
    fuentes = seg["fuente"]
    rng = np.random.default_rng(semilla)
    k = copias(fuentes, repeticiones, rng)
    rondas = int(k.max()) if len(k) else 0
    contenedores = []
    for ronda in range(rondas):
        indices = np.flatnonzero(k > ronda)
        contenedores += [[int(indices[j]) for j in c] for c in bfd(n[indices], max_len)]
    duplicados = sum(1 for c in contenedores if len(c) != len(set(c)))
    if duplicados:
        raise SystemExit(f"{duplicados} contenedores repiten un segmento.")
    (dir_particion / "contenedores.json").write_text(json.dumps(contenedores))
    tokens = Counter()
    for i, veces in enumerate(k):
        tokens[fuentes[i]] += int(n[i]) * int(veces)
    total = sum(tokens.values())
    print(
        f"{dir_particion.name}: contenedores={len(contenedores)} rondas={rondas} "
        f"ocupacion={total / max(len(contenedores) * max_len, 1):.4f}"
    )
    for fuente, t in tokens.most_common():
        print(f"  {fuente:<24} {t:>12} {t / total:.4f}")


class Contenedores(Dataset):
    def __init__(self, dir_particion):
        dir_particion = Path(dir_particion)
        self.seg = load_from_disk(str(dir_particion / "segmentos")).select_columns(["input_ids", "labels"])
        self.contenedores = json.loads((dir_particion / "contenedores.json").read_text())

    def __len__(self):
        return len(self.contenedores)

    def __getitem__(self, i):
        filas = self.seg[self.contenedores[i]]
        return list(zip(filas["input_ids"], filas["labels"]))


class Colador:
    def __init__(self):
        self.aplanar = DataCollatorWithFlattening(
            return_position_ids=True, return_flash_attn_kwargs=True, return_seq_idx=True
        )

    def __call__(self, lote):
        if len(lote) != 1:
            raise ValueError("per_device_train_batch_size debe ser 1: cada elemento ya es un contenedor.")
        return self.aplanar([{"input_ids": x, "labels": y} for x, y in lote[0]])
