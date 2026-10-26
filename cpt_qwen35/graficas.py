import glob
import json
import logging
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROSA = "#DE6F8C"
NARANJA = "#EE9F5C"
TINTA = "#222222"
EJE = "#333333"
MARCAS = "#555555"
REJILLA = "#EEEEEE"
REFERENCIA = "#9A9A9A"
NOTA = 7.5
FASES = {
    "carga_modelo": "Carga del modelo",
    "preparacion": "Preparación",
    "entrenamiento": "Entrenamiento (cómputo)",
    "evaluacion": "Evaluación",
    "sondeo": "Sondeo",
    "guardado": "Checkpoints",
    "cierre": "Cierre",
}

logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

for _fuente in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf"):
    font_manager.fontManager.addfont(_fuente)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.6,
        "axes.labelcolor": TINTA,
        "xtick.color": EJE,
        "ytick.color": EJE,
        "xtick.labelcolor": MARCAS,
        "ytick.labelcolor": MARCAS,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def _jsonl(ruta):
    filas = []
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            try:
                filas.append(json.loads(linea))
            except json.JSONDecodeError:
                pass
    return filas


def leer(dir_metricas):
    d = Path(dir_metricas)
    datos = d / "datos.json"
    return {
        "corrida": json.loads((d / "corrida.json").read_text(encoding="utf-8")),
        "datos": json.loads(datos.read_text(encoding="utf-8")) if datos.exists() else None,
        "pasos": _jsonl(d / "pasos.jsonl"),
        "eventos": _jsonl(d / "eventos.jsonl"),
        "gpu": _jsonl(d / "gpu.jsonl"),
    }


def _entrenamiento(m):
    return [p for p in m["pasos"] if "loss" in p]


def _fin(m):
    t = m["corrida"]["tiempos"]
    if "fin" in t:
        return t["fin"]
    return max([p["t"] for p in m["pasos"]] + [g["t"] for g in m["gpu"]], default=t["inicio"])


def _ventana(m):
    t = m["corrida"]["tiempos"]
    fin = _fin(m)
    return t.get("entrenamiento_inicio", fin), t.get("entrenamiento_fin", fin)


def intervalos(m):
    pasos = sorted(_entrenamiento(m), key=lambda p: p["t"])
    if not pasos:
        return []
    pausas = {p["step"] for p in m["pasos"] if "eval_loss" in p} | {e["step"] for e in m["eventos"]}
    inicio = m["corrida"]["tiempos"].get("entrenamiento_inicio", pasos[0]["t"])
    salida = []
    for a, b in zip(pasos, pasos[1:]):
        dt = b["t"] - a["t"]
        if dt > 0 and "num_input_tokens_seen" in a and "num_input_tokens_seen" in b:
            salida.append(
                {
                    "min": (b["t"] - inicio) / 60,
                    "paso": b["step"],
                    "tok_s": (b["num_input_tokens_seen"] - a["num_input_tokens_seen"]) / dt,
                    "limpio": a["step"] not in pausas,
                }
            )
    return salida


def fases(m):
    t = m["corrida"]["tiempos"]
    fin = _fin(m)
    cargado = t.get("modelo_cargado", fin)
    inicio, fin_ent = _ventana(m)
    evaluacion = sum(p["eval_runtime"] for p in m["pasos"] if "eval_runtime" in p and inicio <= p["t"] <= fin_ent)
    sondeo = sum(e["fin"] - e["inicio"] for e in m["eventos"] if e["evento"] == "sondeo")
    guardado = sum(e["fin"] - e["inicio"] for e in m["eventos"] if e["evento"] == "guardado" and e["inicio"])
    valores = {
        "carga_modelo": cargado - t["proceso"],
        "preparacion": inicio - cargado,
        "entrenamiento": fin_ent - inicio - evaluacion - sondeo - guardado,
        "evaluacion": evaluacion,
        "sondeo": sondeo,
        "guardado": guardado,
        "cierre": fin - fin_ent,
    }
    return {k: max(0.0, v) for k, v in valores.items()}


def energia(m):
    muestras = m["gpu"]
    if not muestras:
        return {}
    por_gpu = []
    for i in range(len(muestras[0]["gpus"])):
        serie = [s["gpus"][i] for s in muestras if i < len(s["gpus"])]
        contador = [g["energia_mj"] for g in serie if g["energia_mj"] is not None]
        potencia = [g["w"] for g in serie if g["w"] is not None]
        vram = [g["vram_mib"] for g in serie if g["vram_mib"] is not None]
        por_gpu.append(
            {
                "energia_kwh": (contador[-1] - contador[0]) / 3.6e9 if len(contador) > 1 else None,
                "potencia_media_w": statistics.fmean(potencia) if potencia else None,
                "vram_pico_mib": max(vram, default=None),
            }
        )
    energias = [g["energia_kwh"] for g in por_gpu if g["energia_kwh"] is not None]
    ram_contenedor = [s["ram_contenedor_mib"] for s in muestras if s.get("ram_contenedor_mib") is not None]
    ram_procesos = [s["ram_procesos_mib"] for s in muestras if s.get("ram_procesos_mib") is not None]
    return {
        "energia_kwh": sum(energias) if energias else None,
        "gpus_medidas": por_gpu,
        "ram_pico_contenedor_mib": max(ram_contenedor, default=None),
        "ram_pico_procesos_mib": max(ram_procesos, default=None),
        "muestras_gpu": len(muestras),
    }


def resumen_datos(d):
    descartados = d.get("descartados")
    salida = {}
    for part, x in d["particiones"].items():
        tokens, con_perdida, contenedores = sum(x["n"]), sum(x["n_perdida"]), x["contenedores"]
        salida[part] = {
            "ejemplos": len(x["n"]),
            "tokens": tokens,
            "fraccion_con_perdida": con_perdida / tokens if tokens else None,
            "contenedores": len(contenedores),
            "ocupacion": sum(contenedores) / (len(contenedores) * d["seq_len"]) if contenedores else None,
            "descartados": sum(1 for r in descartados if r["particion"] == part) if descartados is not None else None,
        }
    return salida


def computo(m, tok_s, tokens):
    n = (m["corrida"].get("modelo") or {}).get("parametros_flops")
    if not n or not tok_s:
        return {}
    return {
        "parametros_flops": n,
        "tflops_logrados": 6 * n * tok_s / 1e12,
        "flops_totales": 6 * n * tokens,
    }


def resumen(m):
    c = m["corrida"]
    e = c["entorno"]
    gpus = e["procesos"]
    duracion = _fin(m) - c["tiempos"]["proceso"]
    limpios = [i["tok_s"] for i in intervalos(m) if i["limpio"]]
    q25, _, q75 = statistics.quantiles(limpios, n=4, method="inclusive") if len(limpios) > 1 else (None, None, None)
    entrenamiento = _entrenamiento(m)
    costo_hora = (e.get("runpod") or {}).get("costo_hora")
    f = fases(m)
    total = sum(f.values())
    tokens = max((p.get("num_input_tokens_seen", 0) for p in m["pasos"]), default=0)
    mediana = statistics.median(limpios) if limpios else None
    return {
        "estado": c.get("estado"),
        "gpu": e["gpus"][0]["nombre"] if e["gpus"] else None,
        "gpus": gpus,
        "duracion_s": duracion,
        "horas_gpu": duracion / 3600 * gpus,
        "costo_usd": duracion / 3600 * costo_hora if costo_hora else None,
        "pasos": c.get("pasos"),
        "pasos_totales": c.get("pasos_totales"),
        "tokens": tokens,
        "tok_s_mediana": mediana,
        "tok_s_q25": q25,
        "tok_s_q75": q75,
        "intervalos_limpios": len(limpios),
        "loss_final": entrenamiento[-1]["loss"] if entrenamiento else None,
        "eval_loss_mejor": c.get("mejor_eval_loss"),
        "mejor_checkpoint": c.get("mejor_checkpoint"),
        "fases_s": f,
        "fraccion_fuera_de_entrenamiento": 1 - f["entrenamiento"] / total if total else None,
        "vram_torch": c.get("vram"),
        **computo(m, mediana, tokens),
        **energia(m),
        "datos": resumen_datos(m["datos"]) if m["datos"] else None,
    }


def _ejes():
    fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=300)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot(1, 0, ">", color=EJE, ms=3.5, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^", color=EJE, ms=3.5, transform=ax.transAxes, clip_on=False)
    return fig, ax


def _guardar(fig, destino, nombre):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(destino / f"{nombre}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return destino / f"{nombre}.png"


def _duracion(segundos):
    if segundos < 90:
        return f"{segundos:.0f} s"
    if segundos < 5400:
        return f"{segundos / 60:.0f} min"
    return f"{segundos / 3600:.2f} h"


def _cuantiles_moviles(valores, ventana):
    mitad = ventana // 2
    salida = np.full((3, valores.size), np.nan)
    for i in range(valores.size):
        tramo = valores[max(0, i - mitad) : i + mitad + 1]
        tramo = tramo[~np.isnan(tramo)]
        if tramo.size:
            salida[:, i] = np.percentile(tramo, (25, 50, 75))
    return salida


def _velocidad(m):
    todos = intervalos(m)
    pasos = np.array([i["paso"] for i in todos])
    tok_s = np.array([i["tok_s"] for i in todos]) / 1e3
    limpio = np.array([i["limpio"] for i in todos], dtype=bool)
    return pasos, tok_s, limpio


def g_longitudes(m, destino):
    d = m["datos"]
    seq_len = d["seq_len"]
    n = np.array([x for p in d["particiones"].values() for x in p["n"]])
    descartados = d.get("descartados")
    largos = np.array([r["n"] for r in descartados or []])
    tope = max(n.max(initial=0), largos.max(initial=0), seq_len) * 1.05
    bins = np.linspace(0, tope, 61)
    fig, ax = _ejes()
    ax.hist(n, bins=bins, color=ROSA, edgecolor="white", linewidth=0.5)
    if largos.size:
        ax.hist(largos, bins=bins, color=REFERENCIA, edgecolor="white", linewidth=0.5)
    ax.axvline(seq_len, color=REFERENCIA, lw=0.8, ls=(0, (4, 3)))
    lineas = [f"seq_len = {seq_len:,}"]
    if descartados is not None:
        lineas.append(f"{largos.size:,} descartados ({largos.size / max(n.size + largos.size, 1):.1%})")
    derecha = seq_len > 0.6 * tope
    texto = "\n".join(f"{l} " if derecha else f" {l}" for l in lineas)
    ax.text(
        seq_len,
        0.97,
        texto,
        transform=ax.get_xaxis_transform(),
        ha="right" if derecha else "left",
        va="top",
        fontsize=NOTA,
        color=MARCAS,
    )
    ax.set_xlim(0, tope)
    ax.set_xlabel("Tokens por segmento")
    ax.set_ylabel("Segmentos")
    return _guardar(fig, destino, "longitudes")


def g_perdida(m, destino):
    entrenamiento = _entrenamiento(m)
    fin_ent = _ventana(m)[1]
    validacion = [p for p in m["pasos"] if "eval_loss" in p and p["t"] <= fin_ent]
    pasos = np.array([p["step"] for p in entrenamiento])
    epocas = np.array([p.get("epoch", 0.0) for p in entrenamiento])
    fig, ax = _ejes()
    ax.plot(pasos, [p["loss"] for p in entrenamiento], color=ROSA, lw=0.7, label="Entrenamiento")
    if validacion:
        ax.plot(
            [p["step"] for p in validacion],
            [p["eval_loss"] for p in validacion],
            color=NARANJA,
            lw=1.4,
            label="Validación",
        )
    for k in range(1, int(np.ceil(epocas.max() - 1e-6))):
        inicio = pasos[np.argmax(epocas > k)]
        ax.axvline(inicio, color=REFERENCIA, lw=0.8, ls=(0, (4, 3)))
        ax.text(
            inicio,
            0.98,
            f" época {k + 1}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=NOTA,
            color=MARCAS,
        )
    if validacion:
        c = m["corrida"]
        if c.get("mejor_checkpoint") and c.get("mejor_eval_loss") is not None:
            paso_mejor = int(Path(c["mejor_checkpoint"]).name.rsplit("-", 1)[-1])
            valor = c["mejor_eval_loss"]
        else:
            mejor = min(validacion, key=lambda p: p["eval_loss"])
            paso_mejor, valor = mejor["step"], mejor["eval_loss"]
        ax.plot(
            paso_mejor,
            valor,
            marker="*",
            ms=10,
            color=NARANJA,
            mec="white",
            mew=0.8,
            zorder=5,
            ls="none",
            label=f"Mejor (paso {paso_mejor}, {valor:.3f})",
        )
    ax.set_xlim(0, pasos[-1])
    ax.set_xlabel("Paso de entrenamiento")
    ax.set_ylabel("Pérdida")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), handlelength=2.0)
    return _guardar(fig, destino, "perdida")


def g_velocidad(m, destino):
    pasos, tok_s, limpio = _velocidad(m)
    serie = np.where(limpio, tok_s, np.nan)
    ventana = max(5, pasos.size // 10)
    p25, p50, p75 = _cuantiles_moviles(serie, ventana)
    mediana = np.nanmedian(serie)
    fig, ax = _ejes()
    ax.fill_between(pasos, p25, p75, color=ROSA, alpha=0.22, linewidth=0, label="Rango intercuartil")
    ax.plot(pasos, p50, color=ROSA, lw=1.6, label=f"Mediana móvil ({ventana} pasos)")
    ax.axhline(mediana, color=REFERENCIA, lw=0.8, ls=(0, (4, 3)), label=f"Mediana total ({mediana:.2f}k)")
    bajo, alto = np.nanpercentile(serie, [1, 99])
    margen = (alto - bajo) * 0.5 or mediana * 0.05
    ax.set_ylim(bajo - margen, alto + margen)
    ax.set_xlim(0, pasos[-1])
    ax.set_xlabel("Paso de entrenamiento")
    ax.set_ylabel(r"Velocidad (tokens/s, $\times 10^3$)")
    ax.legend(loc="upper left", handlelength=2.0)
    return _guardar(fig, destino, "velocidad")


def g_distribucion_velocidad(m, destino):
    _, tok_s, limpio = _velocidad(m)
    limpios = tok_s[limpio]
    q25, mediana, q75 = np.percentile(limpios, [25, 50, 75])
    fig, ax = _ejes()
    alturas, _, _ = ax.hist(
        limpios, bins=min(40, max(10, limpios.size // 10)), color=ROSA, edgecolor="white", linewidth=0.5
    )
    ax.set_ylim(0, alturas.max() * 1.45)
    ax.axvline(mediana, color=REFERENCIA, lw=0.8, ls=(0, (4, 3)))
    ax.text(
        0.97,
        0.95,
        f"mediana {mediana:.2f}k\nRIC {q25:.2f}–{q75:.2f}k\n{int((~limpio).sum())} pasos con pausa excluidos",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=NOTA,
        color=MARCAS,
        linespacing=1.4,
    )
    ax.set_xlabel(r"Velocidad por paso (tokens/s, $\times 10^3$)")
    ax.set_ylabel("Pasos")
    return _guardar(fig, destino, "velocidad_distribucion")


def g_tiempo(m, destino):
    f = {k: v for k, v in fases(m).items() if v > 0}
    total = sum(f.values())
    nombres = sorted(f, key=f.get)
    escala, unidad = (3600, "Horas") if total >= 7200 else (60, "Minutos")
    valores = np.array([f[n] for n in nombres]) / escala
    fig, ax = _ejes()
    ax.grid(False)
    ax.grid(axis="x", color=REJILLA, lw=0.6)
    colores = [ROSA if n == "entrenamiento" else NARANJA for n in nombres]
    ax.barh([FASES[n] for n in nombres], valores, height=0.6, color=colores, linewidth=0)
    for y, n in enumerate(nombres):
        ax.text(
            f[n] / escala,
            y,
            f"  {_duracion(f[n])} · {f[n] / total:.1%}",
            va="center",
            ha="left",
            fontsize=NOTA,
            color=MARCAS,
        )
    ax.tick_params(axis="y", length=0, labelcolor=TINTA)
    ax.set_xlim(0, valores.max() * 1.45)
    fuera = 1 - f.get("entrenamiento", 0.0) / total
    ax.set_xlabel(f"{unidad} (total {_duracion(total)}, {fuera:.0%} sin entrenar)")
    return _guardar(fig, destino, "tiempo")


def generar(dir_metricas):
    m = leer(dir_metricas)
    destino = Path(dir_metricas) / "graficas"
    destino.mkdir(exist_ok=True)
    rutas = []
    if m["datos"]:
        rutas.append(g_longitudes(m, destino))
    if _entrenamiento(m):
        rutas.append(g_perdida(m, destino))
    if sum(i["limpio"] for i in intervalos(m)) >= 2:
        rutas.append(g_velocidad(m, destino))
        rutas.append(g_distribucion_velocidad(m, destino))
    if sum(fases(m).values()) > 0:
        rutas.append(g_tiempo(m, destino))
    return rutas
