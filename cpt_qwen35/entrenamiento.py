import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import Subset
from transformers import Trainer, TrainerCallback, TrainingArguments

from .empaquetado import Colador, Contenedores
from .evaluacion import cargar_lotes, sondear
from .modelo import LIGER, aplicar_lora, cargar, exigir_kernels
from .seguimiento import Seguimiento


class Sondeo(TrainerCallback):
    def __init__(self, lotes, fragmento, destino, registro=None):
        self.lotes = lotes
        self.fragmento = fragmento
        self.destino = Path(destino)
        self.destino.mkdir(parents=True, exist_ok=True)
        self.registro = registro

    def _registrar(self, state, model):
        if not state.is_world_process_zero or not self.lotes:
            return
        inicio = time.time()
        resultado = sondear(model, self.lotes, self.fragmento)
        if self.registro:
            self.registro.anexar(
                "eventos.jsonl", {"evento": "sondeo", "step": state.global_step, "inicio": inicio, "fin": time.time()}
            )
        (self.destino / f"paso-{state.global_step}.json").write_text(json.dumps(resultado, indent=2))
        for nombre, metricas in resultado.items():
            total = metricas["total"]
            print(
                f"[sondeo paso {state.global_step} {nombre}] loss={total['loss']:.4f} "
                f"ganancia={total['ganancia']:.4f} entropia={total['entropia']:.4f} "
                f"kl={total['kl_contra_base']:.5f}"
            )
            for clave, m in sorted(metricas.items()):
                if clave != "total":
                    print(
                        f"    {clave:<28} loss={m['loss']:.4f} ganancia={m['ganancia']:+.4f} "
                        f"kl={m['kl_contra_base']:.4f} tokens={m['tokens']}"
                    )
        if "general" not in resultado:
            print(f"[sondeo paso {state.global_step}] sin lote general: no hay medida de olvido")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._registrar(state, model)

    def on_save(self, args, state, control, model=None, **kwargs):
        self._registrar(state, model)


def subconjunto_eval(val, maximo, semilla):
    if not maximo or maximo >= len(val):
        return val
    rng = np.random.default_rng(semilla)
    return Subset(val, sorted(rng.choice(len(val), maximo, replace=False).tolist()))


def argumentos(cfg, salida, evaluar):
    return TrainingArguments(
        output_dir=salida,
        num_train_epochs=cfg["epocas"],
        max_steps=cfg["max_pasos"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["acumulacion"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["calentamiento"],
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=cfg["estrategia"] if evaluar else "no",
        save_strategy=cfg["estrategia"],
        eval_steps=cfg["pasos_guardado"],
        save_steps=cfg["pasos_guardado"],
        save_total_limit=cfg["checkpoints"],
        load_best_model_at_end=evaluar and cfg.get("mejor_al_final", False),
        metric_for_best_model="eval_loss" if evaluar else None,
        logging_steps=cfg["pasos_registro"],
        include_num_input_tokens_seen="all",
        use_liger_kernel=True,
        liger_kernel_config=LIGER,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        seed=cfg["semilla"],
    )


def entrenar(cfg, salida, reanudar=None):
    signal.signal(signal.SIGTERM, lambda senal, marco: sys.exit(128 + senal))
    seguimiento = Seguimiento(cfg, salida, reanudar)
    try:
        exigir_kernels()
        modelo = aplicar_lora(cargar(cfg["modelo"], liger=True), cfg["lora"])
        seguimiento.modelo_cargado(modelo)
        datos = Path(cfg["datos"]["salida"])
        ent = cfg["entrenamiento"]
        val = Contenedores(datos / "val")
        evaluar = len(val) > 0
        if not evaluar:
            print("sin segmentos de validacion: se entrena sin evaluar")
        sondeo = Sondeo(
            cargar_lotes(cfg["evaluacion"]), cfg["evaluacion"]["fragmento"], Path(salida) / "sondeo", seguimiento.registro
        )
        trainer = Trainer(
            model=modelo,
            args=argumentos(ent, salida, evaluar),
            train_dataset=Contenedores(datos / "train"),
            eval_dataset=subconjunto_eval(val, ent.get("max_contenedores_eval"), ent["semilla"]) if evaluar else None,
            data_collator=Colador(),
            callbacks=[seguimiento.callback, sondeo],
        )
        trainer.train(resume_from_checkpoint=reanudar)
        trainer.save_model(str(Path(salida) / "adaptador_final"))
        if evaluar and not seguimiento.sin_saldo and len(trainer.eval_dataset) < len(val):
            completa = trainer.evaluate(eval_dataset=val)
            print(f"validacion completa: {completa}")
            seguimiento.anotar("validacion_completa", completa)
    except BaseException as e:
        seguimiento.cerrar(e)
        raise
    seguimiento.cerrar()
