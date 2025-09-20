#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] Docker not found in PATH" >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "[ERROR] Docker Compose not found" >&2
  exit 1
fi

COMPOSE_FILE=""
for f in docker-compose.yml compose.yml docker-compose.yaml compose.yaml; do
  if [[ -f "$f" ]]; then COMPOSE_FILE="$f"; break; fi
done
if [[ -z "$COMPOSE_FILE" ]]; then
  echo "[ERROR] No compose file found in $(pwd)" >&2
  exit 1
fi

PROFILE="${1:-all}"
PROFILE_FLAGS=()
case "$PROFILE" in
  cpu) PROFILE_FLAGS=(--profile cpu) ;;
  gpu) PROFILE_FLAGS=(--profile gpu) ;;
  all) PROFILE_FLAGS=(--profile cpu --profile gpu) ;;
  *)   echo "[WARN] Unknown profile '$PROFILE', using 'all'"; PROFILE_FLAGS=(--profile cpu --profile gpu) ;;
esac

"${DC[@]}" -f "$COMPOSE_FILE" "${PROFILE_FLAGS[@]}" build --pull
echo "[OK] Build completed"

