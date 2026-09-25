import json
from pathlib import Path


def benchmarks(cfg, adaptador, destino):
    import lm_eval

    b = cfg["evaluacion"]["benchmarks"]
    variantes = [("base", None)] + ([("ajustado", adaptador)] if adaptador else [])
    resultado = {}
    for etiqueta, peft in variantes:
        args = {"pretrained": cfg["modelo"]["base"], "dtype": "bfloat16"}
        if peft:
            args["peft"] = peft
        r = lm_eval.simple_evaluate(
            model="hf",
            model_args=args,
            tasks=b["tareas"],
            num_fewshot=b["num_fewshot"],
            batch_size=b["batch_size"],
            limit=b["limite"],
        )
        resultado[etiqueta] = r["results"]
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    Path(destino).write_text(json.dumps(resultado, ensure_ascii=False, indent=2, default=str))
    return resultado
