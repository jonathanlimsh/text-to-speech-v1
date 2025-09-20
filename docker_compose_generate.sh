#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load .env if present (same file Compose uses)
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '\n' -I{} echo {}) || true
fi

if [[ -z "${TZ_OFFSET:-}" ]]; then
  TZ_OFFSET="$(date +%z)"
fi
ENV_ARGS=()
if [[ -n "${TZ_OFFSET:-}" ]] ; then
  ENV_ARGS+=(--env TZ_OFFSET="$TZ_OFFSET")
fi

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

SERVICE_IN="${1:-${SERVICE:-auto}}"
if [[ "$SERVICE_IN" != "auto" && "$SERVICE_IN" != "cpu" && "$SERVICE_IN" != "gpu" ]]; then
  SERVICE_IN="auto"
fi

if [[ "$SERVICE_IN" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    SERVICE="tts-gpu"
  else
    SERVICE="tts-cpu"
  fi
elif [[ "$SERVICE_IN" == "gpu" ]]; then
  SERVICE="tts-gpu"
else
  SERVICE="tts-cpu"
fi

# Resolve input dir
INPUT_DIR_VAL="${INPUT_DIR:-}"
if [[ -z "$INPUT_DIR_VAL" ]]; then
  INPUT_DIR_VAL="/app/assets/inputs"
  if [[ -n "${INPUT_SUBDIR:-}" ]]; then
    INPUT_DIR_VAL="/app/assets/inputs/${INPUT_SUBDIR}"
  fi
fi

# Optional second arg overrides input dir (subfolder or absolute / path)
if [[ $# -ge 2 ]]; then
  CAND="$2"
  if [[ "$CAND" != --* ]]; then
    if [[ "$CAND" == /* ]]; then
      INPUT_DIR_VAL="$CAND"
    else
      INPUT_DIR_VAL="/app/assets/inputs/$CAND"
    fi
    shift 2 || true
  else
    shift 1 || true
  fi
else
  shift 1 || true
fi

FORMATS_VAL="${FORMATS:-wav}"
LANGUAGE_ID_VAL="${LANGUAGE_ID:-en}"
DEVICE_VAL="${DEVICE:-auto}"

REC_FLAG=""; [[ "${RECURSIVE:-true}" =~ ^(?i:1|true|yes)$ ]] && REC_FLAG="--recursive"
NOTS_FLAG=""; [[ "${NO_TIMESTAMP:-false}" =~ ^(?i:1|true|yes)$ ]] && NOTS_FLAG="--no-timestamp-dir"
SCH_FLAG=""; [[ "${SPLIT_CHUNKS:-false}" =~ ^(?i:1|true|yes)$ ]] && SCH_FLAG="--split-chunks"
OVR_FLAG="--overwrite"; [[ "${OVERWRITE:-true}" =~ ^(?i:0|false|no)$ ]] && OVR_FLAG=""
NTS_FLAG=""; [[ "${TRIM_SILENCE:-true}" =~ ^(?i:0|false|no)$ ]] && NTS_FLAG="--no-trim-silence"
TTO_FLAG=""; [[ "${TRIM_TAIL_ONLY:-true}" =~ ^(?i:1|true|yes)$ ]] && TTO_FLAG="--trim-tail-only"
PDN_ARGS=(); [[ -n "${PROCESSED_DIR_NAME:-}" ]] && PDN_ARGS=(--processed-dir-name "$PROCESSED_DIR_NAME")
STD_ARGS=(); [[ -n "${SILENCE_TOP_DB:-}" ]] && STD_ARGS=(--silence-top-db "$SILENCE_TOP_DB")
SMD_ARGS=(); [[ -n "${SILENCE_MIN_DUR:-}" ]] && SMD_ARGS=(--silence-min-dur "$SILENCE_MIN_DUR")
SPM_ARGS=(); [[ -n "${SILENCE_PAD_MS:-}" ]] && SPM_ARGS=(--silence-pad-ms "$SILENCE_PAD_MS")

"${DC[@]}" -f "$COMPOSE_FILE" run --rm "$SERVICE" "${ENV_ARGS[@]}" \
  --input-dir "$INPUT_DIR_VAL" \
  --output-dir /app/assets/outputs \
  --formats "$FORMATS_VAL" \
  --language-id "$LANGUAGE_ID_VAL" \
  --device "$DEVICE_VAL" \
  $REC_FLAG $NOTS_FLAG $SCH_FLAG $OVR_FLAG $NTS_FLAG $TTO_FLAG \
  "${PDN_ARGS[@]}" "${STD_ARGS[@]}" "${SMD_ARGS[@]}" "${SPM_ARGS[@]}" \
  --non-interactive

echo "[OK] Generation completed"
