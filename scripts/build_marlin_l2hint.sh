#!/usr/bin/env bash
# Coordinator-only fleet build; also usable with DOCKER_CONTEXT=colima offline.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${L2HINT_BUILD_IMAGE:-sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6}
OUT=${1:?usage: build_marlin_l2hint.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT" ]] || { echo 'Refuse existing output directory' >&2; exit 1; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
DOCKER=(docker)
[[ -z ${DOCKER_CONTEXT:-} ]] || DOCKER+=(--context "$DOCKER_CONTEXT")
ACTUAL=$("${DOCKER[@]}" image inspect "$IMAGE" --format '{{.Id}}')
[[ "$ACTUAL" == sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 ]] || {
  echo 'Pinned image mismatch; no pull or build' >&2; exit 1;
}
# Unique, owned names; preserve stopped containers on success and failure.
TAG="l2hint-$(date +%Y%m%d%H%M%S)-$$"
BUILD_CTN="$TAG-build"
CHECK_CTN="$TAG-selfcheck"
printf '%s\n' "$BUILD_CTN" "$CHECK_CTN" > "$OUT/containers.txt"
cleanup() {
  "${DOCKER[@]}" stop --time 10 "$BUILD_CTN" "$CHECK_CTN" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# No GPU, network, packages, host process namespace, model weights or fleet state.
# Two sequential nvcc calls, each timeout150s; dump calls timeout30s.
"${DOCKER[@]}" run --name "$BUILD_CTN" --network none --cpus 2 --memory 1536m --memory-swap 1536m \
  -v "$ROOT/overlay/bringup:/kit:ro" -v "$OUT:/out" \
  -e PYTHONPATH=/kit -e PYTHONDONTWRITEBYTECODE=1 -e MAX_JOBS=2 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  --entrypoint timeout "$ACTUAL" --signal=TERM --kill-after=10s 480s \
  nice -n 10 python3 -B -m l2hint_native.build --static --out /out \
  > "$OUT/build.log" 2>&1
test -s "$OUT/assembly.json"
# Bind the toolkit stub only in a separate no-GPU registration check.
"${DOCKER[@]}" run --name "$CHECK_CTN" --network none --cpus 2 --memory 1536m --memory-swap 1536m \
  -v "$ROOT/overlay/bringup:/kit:ro" -v "$OUT:/out" \
  -e PYTHONPATH=/kit -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 --entrypoint timeout "$ACTUAL" \
  --signal=TERM --kill-after=10s 45s bash -c \
  'mkdir /out/stubs && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /out/stubs/libcuda.so.1 && export LD_LIBRARY_PATH=/out/stubs:${LD_LIBRARY_PATH:-} && exec nice -n 10 python3 -B -m l2hint_native.selfcheck --out /out' \
  > "$OUT/selfcheck.log" 2>&1
test -s "$OUT/selfcheck.json"
echo "STATIC_PASS $OUT/assembly.json"
