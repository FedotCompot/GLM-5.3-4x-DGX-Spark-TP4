#!/usr/bin/env bash
# Snapshot git reads on the Mac; pinned image has no git and installs nothing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: check_attnfront_cpu_colima.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT" ]] || { echo 'Refuse existing output' >&2; exit 1; }
mkdir -p "$OUT/bin"; OUT=$(cd "$OUT" && pwd)
git -C "$ROOT" archive 261cbdb > "$OUT/base.tar"
git -C "$ROOT" show 261cbdb:overlay/overlay/glm_nvfp4_attn.py > "$OUT/base_apply.py"
shasum -a 256 "$OUT/base.tar" "$OUT/base_apply.py" > "$OUT/base.sha256"
cat > "$OUT/bin/git" <<'PY'
#!/usr/bin/env python3
import pathlib,sys
if sys.argv[1:]==['archive','261cbdb']:
    sys.stdout.buffer.write(pathlib.Path('/receipt/base.tar').read_bytes())
elif sys.argv[1:]==['show','261cbdb:overlay/overlay/glm_nvfp4_attn.py']:
    sys.stdout.buffer.write(pathlib.Path('/receipt/base_apply.py').read_bytes())
else:
    raise SystemExit('snapshot git adapter refuses unsupported command')
PY
chmod +x "$OUT/bin/git"
IMAGE=sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
NAME="attnfront-cpu-$(date +%Y%m%d%H%M%S)-$$"
printf '%s\n' "$NAME" > "$OUT/container-name.txt"
trap 'docker --context colima stop --time 10 "$NAME" >/dev/null 2>&1 || true' EXIT
docker --context colima run --pull never --platform linux/arm64 --name "$NAME" --network none \
 --cpus 2 --memory 1536m --memory-swap 1536m --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges \
 -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
 -e PYTHONPATH=tests -e GLM_MARLIN_L2HINT=off -e GLM_IMAGE_SRC=/usr/local/lib/python3.12/dist-packages \
 -e PYTHONDONTWRITEBYTECODE=1 -e PATH=/receipt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
 -v "$ROOT:/input:ro" -v "$OUT:/receipt:ro" -w /input --entrypoint timeout "$IMAGE" \
 180s nice -n 10 python3 -B -m unittest test_attnfront test_attnfront_host test_attnfront_crossboot test_launcher test_nvfp4_attn \
 > "$OUT/cpu.log" 2>&1
cat "$OUT/cpu.log"
