#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends fonts-lmodern

pip install --ignore-installed cryptography
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y torchvision torchaudio
ARQUITECTURA=$(python -c "import torch; a, b = torch.cuda.get_device_capability(); print(f'{a}{b}')")
FUENTE=$(mktemp -d)
pip download --no-deps --no-binary :all: --no-build-isolation "causal-conv1d==1.7.0" -d "$FUENTE"
tar -xzf "$FUENTE/causal_conv1d-1.7.0.tar.gz" -C "$FUENTE"
python - "$FUENTE/causal_conv1d-1.7.0/setup.py" "$ARQUITECTURA" <<'EOF'
import sys
from pathlib import Path

ruta, arquitectura = Path(sys.argv[1]), sys.argv[2]
texto = ruta.read_text()
original = 'return nvcc_extra_args + ["--threads", "4"]'
if texto.count(original) != 1:
    raise SystemExit("causal-conv1d cambio su setup.py: revisar el filtro de arquitecturas en instalar.sh")
filtro = (
    'return [a for i, a in enumerate(nvcc_extra_args) if a != "-gencode" and (i == 0 or nvcc_extra_args[i - 1] != "-gencode")]'
    f' + ["-gencode", "arch=compute_{arquitectura},code=sm_{arquitectura}", "--threads", "4"]'
)
ruta.write_text(texto.replace(original, filtro))
print(f"causal-conv1d se compila solo para sm_{arquitectura}")
EOF
pip install --no-build-isolation --no-cache-dir "$FUENTE/causal_conv1d-1.7.0"
pip install -e .
python -c "import torch; assert torch.__version__.startswith('2.11.0'), torch.__version__; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
