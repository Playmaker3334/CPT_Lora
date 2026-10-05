import importlib.util
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODULOS_GDN = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
MODULOS_ATN = ("q_proj", "k_proj", "v_proj", "o_proj")
MODULOS_MLP = ("gate_proj", "up_proj", "down_proj")

TARGET = (
    r"^model\.layers\.\d+\.("
    rf"linear_attn\.({'|'.join(MODULOS_GDN)})"
    rf"|self_attn\.({'|'.join(MODULOS_ATN)})"
    rf"|mlp\.({'|'.join(MODULOS_MLP)})"
    r")$"
)

LIGER = {"fused_linear_cross_entropy": True, "rms_norm": False, "swiglu": False}


def exigir_kernels():
    faltan = [p for p in ("fla", "causal_conv1d") if importlib.util.find_spec(p) is None]
    if faltan:
        raise SystemExit(
            f"Faltan {faltan}. El fallback de PyTorch de las capas Gated DeltaNet ignora cu_seqlens y seq_idx: "
            "con empaquetado el estado recurrente pasa de un documento al siguiente."
        )


def tokenizador(cfg):
    return AutoTokenizer.from_pretrained(cfg["base"])


def cargar(cfg, liger):
    if liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

        apply_liger_kernel_to_qwen3_5(**LIGER)
    modelo = AutoModelForCausalLM.from_pretrained(
        cfg["base"],
        dtype=torch.bfloat16,
        attn_implementation=cfg["atencion"],
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    if type(modelo).__name__ != "Qwen3_5ForCausalLM":
        raise SystemExit(f"Se esperaba Qwen3_5ForCausalLM y se obtuvo {type(modelo).__name__}.")
    modelo.config.use_cache = False
    return modelo


def aplicar_lora(modelo, cfg):
    alpha_compuertas = max(1, round(cfg["r_compuertas"] * cfg["alpha"] / cfg["r"]))
    config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET,
        rank_pattern={"in_proj_a": cfg["r_compuertas"], "in_proj_b": cfg["r_compuertas"]},
        alpha_pattern={"in_proj_a": alpha_compuertas, "in_proj_b": alpha_compuertas},
        task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, config)
    tipos = modelo.base_model.model.config.layer_types
    esperado = sum(
        (len(MODULOS_GDN) if t == "linear_attention" else len(MODULOS_ATN)) + len(MODULOS_MLP) for t in tipos
    )
    obtenido = sum(1 for n, _ in modelo.named_modules() if n.endswith(".lora_A"))
    if obtenido != esperado:
        raise SystemExit(f"Adaptadores LoRA: esperados {esperado}, obtenidos {obtenido}.")
    print(f"capas={len(tipos)} gdn={tipos.count('linear_attention')} atencion={tipos.count('full_attention')} adaptadores={obtenido}")
    modelo.print_trainable_parameters()
    return modelo


def cargar_para_evaluar(cfg, adaptador=None):
    modelo = cargar(cfg, liger=False)
    if adaptador:
        modelo = PeftModel.from_pretrained(modelo, adaptador)
    return modelo.eval()


def fusionar(cfg, adaptador, destino):
    modelo = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.bfloat16, device_map="cpu")
    modelo = PeftModel.from_pretrained(modelo, adaptador).merge_and_unload()
    modelo.save_pretrained(destino)
    tokenizador(cfg).save_pretrained(destino)
