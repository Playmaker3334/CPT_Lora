import argparse
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preparar")
    sp.add_argument("--retokenizar", action="store_true")

    sub.add_parser("verificar")

    sp = sub.add_parser("entrenar")
    sp.add_argument("--salida", required=True)
    sp.add_argument("--reanudar", default=None)

    sp = sub.add_parser("evaluar")
    sp.add_argument("--adaptador", default=None)
    sp.add_argument("--salida", required=True)
    sp.add_argument("--benchmarks", action="store_true")

    sp = sub.add_parser("fusionar")
    sp.add_argument("--adaptador", required=True)
    sp.add_argument("--salida", required=True)

    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))

    if a.cmd == "preparar":
        from cpt_qwen35.corpus import documentos
        from cpt_qwen35.empaquetado import empaquetar
        from cpt_qwen35.evaluacion import construir_lotes
        from cpt_qwen35.modelo import tokenizador
        from cpt_qwen35.tokenizacion import construir

        d = cfg["datos"]
        salida = Path(d["salida"])
        if a.retokenizar or not (salida / "train" / "segmentos").exists():
            construir(documentos(d), tokenizador(cfg["modelo"]), d, salida)
        semilla = cfg["entrenamiento"]["semilla"]
        empaquetar(salida / "train", d["seq_len"], cfg["mezcla"]["repeticiones"], semilla)
        empaquetar(salida / "val", d["seq_len"], None, semilla)
        construir_lotes(cfg["evaluacion"], salida, semilla)

    elif a.cmd == "verificar":
        from cpt_qwen35.verificacion import verificar

        verificar(cfg)

    elif a.cmd == "entrenar":
        from cpt_qwen35.entrenamiento import entrenar

        entrenar(cfg, a.salida, a.reanudar)

    elif a.cmd == "evaluar":
        from cpt_qwen35.evaluacion import evaluar
        from cpt_qwen35.modelo import cargar_para_evaluar, exigir_kernels, tokenizador

        exigir_kernels()
        salida = Path(a.salida)
        modelo = cargar_para_evaluar(cfg["modelo"], a.adaptador)
        evaluar(modelo, tokenizador(cfg["modelo"]), cfg["evaluacion"], salida / "evaluacion.json")
        if a.benchmarks:
            del modelo
            from cpt_qwen35.benchmarks import benchmarks

            benchmarks(cfg, a.adaptador, salida / "benchmarks.json")

    elif a.cmd == "fusionar":
        from cpt_qwen35.modelo import fusionar

        fusionar(cfg["modelo"], a.adaptador, a.salida)


if __name__ == "__main__":
    main()
