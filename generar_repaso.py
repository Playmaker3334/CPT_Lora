import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="Genera el JSONL de repaso en español a partir de FineWeb-2 (spa_Latn).")
    p.add_argument("destino")
    p.add_argument("tokens", type=int, help="Tokens objetivo; para un repaso del 10%% usar tokens_juridicos / 9.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--semilla", type=int, default=0)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["modelo"]["base"])
    flujo = load_dataset("HuggingFaceFW/fineweb-2", "spa_Latn", split="train", streaming=True)
    flujo = flujo.shuffle(seed=a.semilla, buffer_size=10_000)

    destino = Path(a.destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    total = docs = 0
    with open(destino, "w", encoding="utf-8") as f:
        for fila in flujo:
            texto = fila["text"].strip()
            if not texto:
                continue
            total += len(tok(texto, add_special_tokens=False)["input_ids"])
            docs += 1
            f.write(json.dumps({"texto": texto}, ensure_ascii=False) + "\n")
            if total >= a.tokens:
                break
    print(f"documentos={docs} tokens={total}")


if __name__ == "__main__":
    main()
