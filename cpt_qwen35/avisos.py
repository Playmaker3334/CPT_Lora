import os
import time
from pathlib import Path

from slack_sdk import WebClient

from .graficas import FASES

ESTADOS = {
    "en_curso": "en curso (reporte parcial)",
    "detenido_por_saldo": "se detuvo porque se acababa el saldo de RunPod (adaptador parcial guardado)",
    "terminado": "terminó",
    "fallo": "FALLÓ",
    "interrumpido": "fue interrumpido",
    "detenido_loss_no_finita": "se detuvo: loss no finita",
}


class Slack:
    def __init__(self, hilo=None, canal=None):
        token = os.environ.get("SLACK_BOT_TOKEN")
        self.canal = canal or os.environ.get("SLACK_CANAL")
        self.cliente = WebClient(token=token) if token and self.canal else None
        self.hilo = hilo
        if self.cliente is None:
            print("slack: SLACK_BOT_TOKEN o SLACK_CANAL sin definir; no se envian avisos")

    def mensaje(self, texto):
        if self.cliente is None:
            return
        try:
            respuesta = self.cliente.chat_postMessage(channel=self.canal, text=texto, thread_ts=self.hilo)
            self.canal = respuesta["channel"]
            self.hilo = self.hilo or respuesta["ts"]
        except Exception as e:
            print(f"slack: {e}")

    def imagenes(self, rutas, texto):
        if self.cliente is None or not rutas:
            return
        try:
            self.cliente.files_upload_v2(
                channel=self.canal,
                thread_ts=self.hilo,
                initial_comment=texto,
                file_uploads=[{"file": str(r), "title": r.stem} for r in rutas],
            )
        except Exception as e:
            print(f"slack: {e}")


class Latido:
    def __init__(self, slack, nombre, estado):
        self.slack = slack
        self.nombre = nombre
        self.estado = estado
        self.t0 = time.time()
        self.previo = None

    def __call__(self):
        e = dict(self.estado)
        ahora = time.time()
        partes = [self.nombre, f"{(ahora - self.t0) / 60:.0f} min"]
        if "paso" not in e:
            partes.append("aún sin pasos registrados")
        else:
            partes += [f"paso {e['paso']}/{e.get('max_pasos', '?')}", f"loss {e['loss']:.4f}"]
            p = self.previo
            if p and e["t"] > p["t"] and e.get("tokens") is not None and p.get("tokens") is not None:
                partes.append(f"{(e['tokens'] - p['tokens']) / (e['t'] - p['t']):,.0f} tok/s promedio desde el aviso anterior")
            if ahora - e["t"] > 300:
                partes.append(f"sin logs desde hace {(ahora - e['t']) / 60:.0f} min")
            self.previo = e
        if "saldo_min" in e:
            partes.append(f"saldo para ~{e['saldo_min']:.0f} min")
        self.slack.mensaje(" · ".join(partes))


def _gib(mib):
    return f"{mib / 1024:.1f} GiB" if mib is not None else "s/d"


def texto_arranque(c, datos):
    e = c["entorno"]
    cfg = c["config"]
    gpu = e["gpus"][0]["nombre"] if e["gpus"] else "GPU"
    git = e.get("git") or {}
    commit = (git.get("commit") or "s/d")[:7] + (" con cambios sin commit" if git.get("cambios_sin_commit") else "")
    lineas = [f"*{c['nombre']}* arrancó", f"{e['procesos']}× {gpu} · commit {commit}"]
    for part, d in datos.items():
        if not d["ejemplos"]:
            lineas.append(f"{part}: sin segmentos")
            continue
        descartados = f" · {d['descartados']:,} descartados por exceder seq_len" if d.get("descartados") else ""
        lineas.append(
            f"{part}: {d['ejemplos']:,} segmentos · {d['tokens']:,} tokens · "
            f"{d['fraccion_con_perdida']:.0%} con pérdida · ocupación de contenedores {d['ocupacion']:.0%}{descartados}"
        )
    ent = cfg["entrenamiento"]
    epocas = f"{ent['epocas']} {'época' if ent['epocas'] == 1 else 'épocas'}"
    lineas.append(f"{epocas} · lr {ent['lr']} · LoRA r={cfg['lora']['r']} · seq_len {cfg['datos']['seq_len']:,}")
    return "\n".join(lineas)


def texto_resumen(c):
    r = c["resumen"]
    lineas = [f"*{c['nombre']}* {ESTADOS.get(r['estado'], r['estado'])}"]
    if "error" in c:
        lineas.append(f"```{c['error']['traza'][-1500:]}```")
    costo = f" · costo ${r['costo_usd']:.2f}" if r.get("costo_usd") is not None else ""
    lineas.append(f"Duración {r['duracion_s'] / 60:.1f} min · {r['horas_gpu']:.2f} horas-GPU{costo}")
    if r.get("pasos") is not None:
        lineas.append(f"Pasos {r['pasos']}/{r['pasos_totales']}")
    velocidad = ""
    if r.get("tok_s_mediana") is not None:
        velocidad = f" · mediana {r['tok_s_mediana']:,.0f} tok/s"
        if r.get("tok_s_q25") is not None:
            velocidad += f" (RIC {r['tok_s_q25']:,.0f}–{r['tok_s_q75']:,.0f})"
    lineas.append(f"Tokens {r['tokens']:,}{velocidad}")
    if r.get("tflops_logrados") is not None:
        lineas.append(
            f"Cómputo (aprox. 6N): {r['tflops_logrados']:.1f} TFLOPS · {r['flops_totales']:.2e} FLOPs totales"
        )
    if r.get("energia_kwh") is not None:
        potencias = " / ".join(f"{g['potencia_media_w']:.0f}" for g in r["gpus_medidas"] if g["potencia_media_w"] is not None)
        lineas.append(f"Energía de las GPUs {r['energia_kwh']:.3f} kWh · potencia media {potencias} W")
    if r.get("vram_torch"):
        lineas.append("VRAM pico reservada por PyTorch: " + " / ".join(_gib(v["reservada_mib"]) for v in r["vram_torch"]))
    if r.get("ram_pico_contenedor_mib") is not None or r.get("ram_pico_procesos_mib") is not None:
        lineas.append(
            f"RAM pico: contenedor {_gib(r.get('ram_pico_contenedor_mib'))} · "
            f"procesos {_gib(r.get('ram_pico_procesos_mib'))}"
        )
    if r.get("loss_final") is not None:
        mejor = ""
        if r.get("eval_loss_mejor") is not None and r.get("mejor_checkpoint"):
            mejor = f" · mejor validación {r['eval_loss_mejor']:.4f} ({Path(r['mejor_checkpoint']).name})"
        lineas.append(f"Loss final {r['loss_final']:.4f}{mejor}")
    if r.get("fraccion_fuera_de_entrenamiento") is not None:
        otras = ", ".join(
            f"{FASES[k].lower()} {v / 60:.1f} min" for k, v in r["fases_s"].items() if k != "entrenamiento" and v >= 1
        )
        lineas.append(f"Fuera de entrenamiento: {r['fraccion_fuera_de_entrenamiento']:.0%} ({otras})")
    return "\n".join(lineas)
