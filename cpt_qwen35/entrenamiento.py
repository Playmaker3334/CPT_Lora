import json
from pathlib import Path

from transformers import Trainer, TrainerCallback, TrainingArguments

from .empaquetado import Colador, Contenedores
from .evaluacion import cargar_lotes, sondear
from .modelo import aplicar_lora, cargar, exigir_kernels


class Sondeo(TrainerCallback):
    def __init__(self, lotes, fragmento, destino):
        self.lotes = lotes
        self.fragmento = fragmento
        self.destino = Path(destino)
        self.destino.mkdir(parents=True, exist_ok=True)

    def _registrar(self, state, model):
        if not state.is_world_process_zero or not self.lotes:
            return
        resultado = sondear(model, self.lotes, self.fragmento)
        (self.destino / f"paso-{state.global_step}.json").write_text(json.dumps(resultado, indent=2))
        total = resultado.get("juridico", {}).get("total")
        if total:
            print(
                f"[sondeo paso {state.global_step}] loss={total['loss']:.4f} ganancia={total['ganancia']:.4f} "
                f"entropia={total['entropia']:.4f} kl={total['kl_contra_base']:.5f}"
            )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._registrar(state, model)

    def on_save(self, args, state, control, model=None, **kwargs):
        self._registrar(state, model)


def argumentos(cfg, salida):
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
        eval_strategy=cfg["estrategia"],
        save_strategy=cfg["estrategia"],
        eval_steps=cfg["pasos_guardado"],
        save_steps=cfg["pasos_guardado"],
        logging_steps=cfg["pasos_registro"],
        include_num_input_tokens_seen="all",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        seed=cfg["semilla"],
    )


def entrenar(cfg, salida, reanudar=None):
    exigir_kernels()
    modelo = aplicar_lora(cargar(cfg["modelo"], liger=True), cfg["lora"])
    datos = Path(cfg["datos"]["salida"])
    trainer = Trainer(
        model=modelo,
        args=argumentos(cfg["entrenamiento"], salida),
        train_dataset=Contenedores(datos / "train"),
        eval_dataset=Contenedores(datos / "val"),
        data_collator=Colador(),
        callbacks=[Sondeo(cargar_lotes(cfg["evaluacion"]), cfg["evaluacion"]["fragmento"], Path(salida) / "sondeo")],
    )
    trainer.train(resume_from_checkpoint=reanudar)
    trainer.save_model(str(Path(salida) / "adaptador_final"))
