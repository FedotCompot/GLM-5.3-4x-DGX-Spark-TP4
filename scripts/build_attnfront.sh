#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: build_attnfront.sh NEW_RECEIPT_DIRECTORY}
mkdir -p "$OUT"; chmod 0777 "$OUT"; OUT=$(cd "$OUT" && pwd)
[[ ! -e "$OUT/build.json" ]] || { echo 'Refuse existing receipt' >&2; exit 1; }
IMAGE=sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
CTX=${ATTNFRONT_DOCKER_CONTEXT:-default}
NAME="attnfront-build-$(date +%Y%m%d%H%M%S)-$$"
printf '%s\n' "$NAME" > "$OUT/container-name.txt"
trap 'docker --context "$CTX" stop --time 10 "$NAME" >/dev/null 2>&1 || true' EXIT
[[ $(docker --context "$CTX" image inspect "$IMAGE" --format '{{.Architecture}}') == arm64 ]]
docker --context "$CTX" run --pull never --platform linux/arm64 --name "$NAME" --network none \
 --cpus 2 --memory 1536m --memory-swap 1536m --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges \
 -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e MAX_JOBS=2 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
 -v "$ROOT:/input:ro" -v "$OUT:/out" --entrypoint timeout "$IMAGE" --signal=TERM --kill-after=10s 600s \
 nice -n 10 python3 -B /input/overlay/bringup/attnfront_native/build.py --out /out \
 > "$OUT/build.log" 2>&1
cat "$OUT/build.log"
mkdir -p "$ROOT/overlay/bringup/attnfront_native/build"
cp "$OUT/attnfront.so" "$OUT/build.json" "$ROOT/overlay/bringup/attnfront_native/build/"
