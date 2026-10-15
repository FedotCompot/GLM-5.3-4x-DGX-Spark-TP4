#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS_ROOT=${1:-$HOME/models}
case "${2:-stock}" in
  stock) python3 "$ROOT/scripts/weights.py" download target "$MODELS_ROOT/Tech2wild/GLM-5.3-Int4-Int8Mix" ;;
  abliterated) python3 "$ROOT/scripts/weights.py" download abliterated "$MODELS_ROOT/Blackfrost-AI/GLM-5.3-DERISKED-Int4-Int8Mix" ;;
  *) echo 'usage: download_weights.sh [MODELS_ROOT] [stock|abliterated]' >&2; exit 1 ;;
esac
# Optional historical DSpark fallback only:
# python3 "$ROOT/scripts/weights.py" download drafter "$MODELS_ROOT/RedHatAI/GLM-5.3-speculator.dspark"
