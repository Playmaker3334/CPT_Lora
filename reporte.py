import argparse
import ast
import json
import re
from pathlib import Path


def sondeos(salida):
    archivos = sorted(Path(salida, "sondeo").glob("paso-*.json"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    return [(int(re.search(r"\d+", p.stem).group()), json.loads(p.read_text(encoding="utf-8"))) for p in archivos]


def tabla_evolucion(pasos):
    print("=== evolucion del sondeo: loss | ganancia contra el modelo base ===")
    print(f"{'lote':<9} {'clave':<28} {'tokens':>7}  " + "  ".join(f"{'paso ' + str(p):>15}" for p, _ in pasos))
    ultimo = pasos[-1][1]
    for lote in ultimo:
        for clave in sorted(ultimo[lote]):
            celdas = []
            for _, r in pasos:
                m = r.get(lote, {}).get(clave)
                celdas.append(f"{m['loss']:.3f}|{m['ganancia']:+.3f}" if m else f"{'-':>15}")
            print(f"{lote:<9} {clave:<28} {ultimo[lote][clave]['tokens']:>7}  " + "  ".join(f"{c:>15}" for c in celdas))


def tabla_final(paso, r):
    print(f"\n=== modelo base contra ajustado en el paso {paso} ===")
    print(f"{'lote':<9} {'clave':<28} {'loss_base':>9} {'loss_ajus':>9} {'mejora':>7} {'kl':>7}")
    for lote in r:
        for clave, m in sorted(r[lote].items()):
            mejora = m["ganancia"] / m["loss_base"] if m["loss_base"] else 0.0
            print(f"{lote:<9} {clave:<28} {m['loss_base']:>9.3f} {m['loss']:>9.3f} {mejora:>7.1%} {m['kl_contra_base']:>7.3f}")


def tabla_benchmarks(ruta):
    r = json.loads(ruta.read_text(encoding="utf-8"))
    if "ajustado" not in r:
        return
    print(f"\n=== benchmarks ({ruta}) ===")
    print(f"{'tarea':<20} {'metrica':<14} {'base':>7} {'ajustado':>8} {'dif':>8} {'margen95':>9}  significativa")
    for tarea, base in r["base"].items():
        ajustado = r["ajustado"].get(tarea, {})
        for metrica in ("acc,none", "acc_norm,none"):
            if metrica in base and metrica in ajustado:
                dif = ajustado[metrica] - base[metrica]
                margen = 2 * base.get(metrica.replace(",", "_stderr,"), 0.0)
                print(
                    f"{tarea:<20} {metrica:<14} {base[metrica]:>7.4f} {ajustado[metrica]:>8.4f} {dif:>+8.4f} "
                    f"{margen:>9.4f}  {'si' if abs(dif) > margen else 'no'}"
                )


def curva(log, bloque):
    perdidas = []
    for linea in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        i = linea.find("{'loss'")
        if i < 0:
            continue
        try:
            perdidas.append(float(ast.literal_eval(linea[i : linea.rindex("}") + 1])["loss"]))
        except (ValueError, SyntaxError):
            continue
    print(f"\n=== loss de entrenamiento, promedio por bloques de {bloque} pasos ({len(perdidas)} pasos) ===")
    for i in range(0, len(perdidas), bloque):
        b = perdidas[i : i + bloque]
        print(f"  pasos {i + 1:>5}-{i + len(b):<5} {sum(b) / len(b):.4f}")


def main():
    p = argparse.ArgumentParser(description="Reporte de una corrida: sondeo, benchmarks y curva de loss.")
    p.add_argument("salida", help="Carpeta de la corrida, ej. salidas/cpt")
    p.add_argument("--log", default=None)
    p.add_argument("--bloque", type=int, default=25)
    a = p.parse_args()

    pasos = sondeos(a.salida)
    if pasos:
        tabla_evolucion(pasos)
        tabla_final(*pasos[-1])
    else:
        print(f"sin sondeos en {a.salida}/sondeo")
    for ruta in sorted(Path(a.salida).rglob("benchmarks.json")):
        tabla_benchmarks(ruta)
    if a.log:
        curva(a.log, a.bloque)


if __name__ == "__main__":
    main()
