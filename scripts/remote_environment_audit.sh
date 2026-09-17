#!/usr/bin/env bash
set -euo pipefail
HOST_NAME="$1"; PORT="$2"; OUT="$3"
mkdir -p "$OUT"
ssh -p "$PORT" -o StrictHostKeyChecking=no root@connect.westc.seetacloud.com 'printf "host=%s\n" "$(hostname)"; printf "pwd=%s\n" "$PWD"; python -V 2>&1; command -v python; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 || true; find / -maxdepth 4 -type d -name .git 2>/dev/null | head -50; find / -maxdepth 4 -type d -iname "*data*" 2>/dev/null | head -50; for d in $(find / -maxdepth 4 -type d -name .git 2>/dev/null | head -20); do git -C "$d/.." rev-parse HEAD 2>/dev/null | sed "s#^#git=$d/.. #"; done' | tee "$OUT/${HOST_NAME}_environment.txt"
sha256sum "$OUT/${HOST_NAME}_environment.txt" > "$OUT/${HOST_NAME}_environment.sha256"
