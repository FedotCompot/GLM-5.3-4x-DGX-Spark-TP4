#!/usr/bin/env bash
# Native GB10 kit, constrained to common-rules workstation limits.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${GB10CHECK_DIR:?Set GB10CHECK_DIR to the saved GB10 kit}"
OUT=${1:?usage: check_attnfront_colima.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT" ]] || { echo 'Refuse existing output' >&2; exit 1; }
mkdir -p "$OUT/package/overlay/overlay" "$OUT/package/overlay/bringup" "$OUT/package/bench/attnfront_static"
OUT=$(cd "$OUT" && pwd)
cp "$ROOT/overlay/overlay/glm_nvfp4_attn.py" "$OUT/package/overlay/overlay/"
cp -R "$ROOT/overlay/bringup/attnfront_native" "$OUT/package/overlay/bringup/"
cp "$ROOT/overlay/bringup/"{glm_attnfront.py,glm_attnfront_kernel.py,glm_fp4_kv_kernel.py,attnfront_source_pins.json,attnfront_local_pins.json} "$OUT/package/overlay/bringup/"
cp "$ROOT/bench/attnfront_static/compile.py" "$OUT/package/bench/attnfront_static/"
cp "$ROOT/bench/attnfront_static/test_static.py" "$OUT/package/"
python3 - "$OUT/manifest.json" <<'PY'
import json,sys
triton=[]
for dtype in ('bf16','fp32'):
 for rs in (576,2112):
  for neox in (False,True):
   for pack in (False,True):
    triton.append(dict(name=f'rope-pack-{dtype}-rs{rs}-neox{int(neox)}-pack{int(pack)}',
      file='overlay/bringup/glm_attnfront_kernel.py',function='_rope_pack',
      signature=dict(Q='*bf16',R='*bf16',X='*bf16',C='*'+dtype,P='*i64',KV='*u8',SLOTS='*i64',SCALE='*fp32',N='i32',NS='i32'),
      constants=dict(QS=4096,QH=256,RS=rs,XS=512,KS=368,H=16,NEOX=neox,PACK=pack),
      options=dict(num_warps=4,num_stages=1,enable_fp_fusion=False)))
json.dump(dict(triton=triton,commands=[dict(name='dual-rms',argv=['python3','-B','{package}/bench/attnfront_static/compile.py','{out}'],binary='{out}/native/attnfront.so',threads=192,dynamic_shared=0)]),open(sys.argv[1],'w'),indent=2)
PY
IMAGE=sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
NAME="gb10check-attnfront-$(date +%Y%m%d%H%M%S)-$$"
printf '%s\n' "$NAME" > "$OUT/container-name.txt"
trap 'docker --context colima stop --time 10 "$NAME" >/dev/null 2>&1 || true' EXIT
WORKSPACE=$(cd "$ROOT/../../../.." && pwd)
docker --context colima run --pull never --platform linux/arm64 --name "$NAME" --network none \
 --cpus 2 --memory 1536m --memory-swap 1536m --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges \
 -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e MAX_JOBS=2 \
 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e GB10_IMAGE="$IMAGE" \
 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work/Projects/sparks/diagnostics/glm53-full-20260929/day2/attnfront/overlay/bringup \
 -v "$GB10CHECK_DIR:/kit:ro" -v "$OUT/package:/input:ro" -v "$OUT:/out" \
 -v "$OUT/manifest.json:/manifest.json:ro" -v "$WORKSPACE:/workspace:ro" \
 --entrypoint timeout "$IMAGE" --signal=TERM --kill-after=10s 840s \
 nice -n 10 bash -c 'set -e
python3 -B /input/overlay/bringup/attnfront_native/build.py --out /out/native > /out/native-build.log 2>&1
python3 -B /kit/check.py --profile attnfront --manifest /manifest.json
' > "$OUT/run.log" 2>&1
cat "$OUT/run.log"
