#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export STUDIO_DATA_DIR="${STUDIO_DATA_DIR:-$(pwd)/data}"
export EVOLVE_STUDIO_PORT="${EVOLVE_STUDIO_PORT:-8771}"
export EVOLVE_STUDIO_HOST="${EVOLVE_STUDIO_HOST:-0.0.0.0}"
# load .env
if [[ -f .env ]]; then set -a; source .env; set +a; fi
exec python3 server.py
