# Text-to-Speech CLI (Chatterbox)

Generate audio from a folder of text files using Resemble AI's Chatterbox TTS. The CLI prompts for configuration (folders, formats, language, optional reference audio, and model params) and produces WAV/MP3/OGG files. Docker is provided so you donâ€™t need local pip installs.

## Quick Start (Docker)

- CPU build:
  docker build -t tts-cli:cpu text-to-speech-v1

- Run interactively (mount assets so outputs persist):
  # Windows PowerShell
  docker run --rm -it -v ${PWD}/text-to-speech-v1/assets:/app/assets tts-cli:cpu
  # macOS/Linux
  # docker run --rm -it -v "$(pwd)/text-to-speech-v1/assets:/app/assets" tts-cli:cpu

- Non-interactive example:
  # Windows PowerShell
  docker run --rm -it -v ${PWD}/text-to-speech-v1/assets:/app/assets \
    tts-cli:cpu \
    --input-dir /app/assets/inputs \
    --output-dir /app/assets/outputs \
    --formats wav,mp3 \
    --language-id en \
    --temperature 0.8 \
    --exaggeration 0.5 \
    --cfg-weight 0.5 \
    --seed 0 \
    --device cpu \
    --overwrite --non-interactive

## GPU with Docker Compose

Prerequisites: NVIDIA GPU + driver on host and NVIDIA Container Toolkit installed.

- Build and run CPU via compose:
  docker compose -f text-to-speech-v1/docker-compose.yml --profile cpu up tts-cpu

- Build and run GPU via compose:
  docker compose -f text-to-speech-v1/docker-compose.yml --profile gpu up tts-gpu

- Run a one-off GPU command (non-interactive):
  docker compose -f text-to-speech-v1/docker-compose.yml --profile gpu run --rm tts-gpu \
    --input-dir /app/assets/inputs --output-dir /app/assets/outputs --formats wav,mp3 \
    --language-id en --temperature 0.8 --exaggeration 0.5 --cfg-weight 0.5 \
    --seed 0 --device cuda --overwrite --non-interactive

### Docker Compose helper scripts (Windows)

- Build images (default builds both cpu and gpu profiles):
  - `text-to-speech-v1/docker_compose_build.bat` [cpu|gpu|all]

- Apply release (pull latest, build with latest bases, and start services):
  - `text-to-speech-v1/docker_compose_release.bat` [auto|cpu|gpu|all]
    - `auto` picks `gpu` if `nvidia-smi` is available, else `cpu`.
    - Uses the compose file in `text-to-speech-v1/` (docker-compose.yml/compose.yml).

### Generate once via Docker Compose (no batch)

- GPU (if available):
  docker compose -f text-to-speech-v1/docker-compose.yml run --rm tts-gpu \
    --input-dir /app/assets/inputs \
    --output-dir /app/assets/outputs \
    --formats wav,ogg \
    --language-id en \
    --temperature 0.8 --exaggeration 0.5 --cfg-weight 0.5 \
    --seed 0 --device auto \
    --overwrite --non-interactive

- CPU:
  docker compose -f text-to-speech-v1/docker-compose.yml run --rm tts-cpu \
    --input-dir /app/assets/inputs \
    --output-dir /app/assets/outputs \
    --formats wav \
    --language-id en \
    --seed 0 --device cpu \
    --overwrite --non-interactive

Note: The images define an ENTRYPOINT that already runs the CLI. Do not prefix `python cli.py`; pass flags only.

You can append any CLI flags here (e.g., `--recursive`, `--no-timestamp-dir`, `--split-chunks`, `--silence-top-db 35`).

#### Examples with recursive processing and auto-detected language

- GPU, scan subfolders, trim silence (default), single WAV per text:
  docker compose -f text-to-speech-v1/docker-compose.yml run --rm tts-gpu \
    --input-dir /app/assets/inputs \
    --output-dir /app/assets/outputs \
    --formats wav \
    --device auto --recursive --overwrite --non-interactive

- CPU, keep per-chunk files and disable timestamped folder:
  docker compose -f text-to-speech-v1/docker-compose.yml run --rm tts-cpu \
    --input-dir /app/assets/inputs \
    --output-dir /app/assets/outputs \
    --formats wav,ogg \
    --split-chunks --no-timestamp-dir --recursive \
    --device cpu --overwrite --non-interactive

### Generate via helper scripts

- Windows: `text-to-speech-v1/docker_compose_generate.bat` [auto|cpu|gpu] [optional input subfolder]
  - Uses a single `.env` file for configuration shared with Docker Compose.
  - Edit `text-to-speech-v1/.env` (copy from `.env.example`) to set:
    - `SERVICE=auto|cpu|gpu`, `INPUT_SUBDIR=YourFolder`, `FORMATS=wav,ogg`, `RECURSIVE=true`, etc.
    - Leave `TZ_OFFSET` blank to auto-detect your host offset (scripts export it for the container).
  - Run:
    - `text-to-speech-v1\docker_compose_generate.bat` (uses `.env`)
    - or override subfolder: `text-to-speech-v1\docker_compose_generate.bat auto Your_Subfolder`

- macOS/Linux: `text-to-speech-v1/docker_compose_generate.sh` [auto|cpu|gpu] [optional input subfolder]
  - Also reads `text-to-speech-v1/.env` for defaults (same keys as above).
  - Example:
    - `./text-to-speech-v1/docker_compose_generate.sh` (uses `.env`)
    - `./text-to-speech-v1/docker_compose_generate.sh cpu MySubfolder`

Note: The compose file maps `./assets` into `/app/assets`. Model downloads persist in Docker layer cache; outputs persist in the mapped assets/outputs folder.

## Beginner Guide (Windows, Docker Desktop)

This section is for users new to Docker who want to build and run the service endâ€‘toâ€‘end and generate TTS outputs.

1) Install prerequisites

- Docker Desktop for Windows: https://docs.docker.com/desktop/install/windows/
- Windows features: ensure WSL 2 and Virtualization are enabled (Docker Desktop installer guides you).
- Optional GPU acceleration: install the latest NVIDIA driver. In Docker Desktop Settings, enable the WSL 2 based engine and GPU support. Verify GPU with:
  - `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

2) Prepare inputs/outputs

- Place `.txt` files under `text-to-speech-v1/assets/inputs` (you can create subfolders).
- Outputs appear under `text-to-speech-v1/assets/outputs/<timestamp>/<subfolder>/<lang>/...`.

3) Build images (first time, or after code changes)

- Doubleâ€‘click `text-to-speech-v1/docker_compose_build.bat` (or run it in a terminal).

4) Configure once via `.env`

- Copy `text-to-speech-v1/.env.example` to `text-to-speech-v1/.env` and edit keys:
  - `SERVICE=auto` (or `cpu`/`gpu`)
  - `INPUT_SUBDIR=` (optional: a subfolder under `assets/inputs`)
  - `FORMATS=wav,ogg`, `RECURSIVE=true`, `NO_TIMESTAMP=false`, etc.
  - Silence trimming knobs: `TRIM_SILENCE`, `TRIM_TAIL_ONLY`, `SILENCE_TOP_DB`, `SILENCE_MIN_DUR`, `SILENCE_PAD_MS`
  - Optional: `HF_TOKEN=...` if you need private model access (Compose already forwards it)

5) Generate audio

- Oneâ€‘shot (no background containers):
  - All inputs recursively: `text-to-speech-v1\docker_compose_generate.bat auto`
  - Specific subfolder: `text-to-speech-v1\docker_compose_generate.bat auto Your_Subfolder_Name`
    - `auto` picks GPU if available, else CPU. Use `cpu` or `gpu` to force.

- Or start services (release):
  - `text-to-speech-v1\docker_compose_release.bat auto`
  - Then you can use `docker compose run --rm tts-gpu --help` etc.

6) Customize behavior

- Prefer editing `text-to-speech-v1/.env` so both Docker Compose and helper scripts stay in sync.
- You can still pass a subfolder on the command to override `INPUT_SUBDIR` for a oneâ€‘off run.

7) Environment variables (optional)

- Public models need no secrets. For private models on Hugging Face, set `HF_TOKEN` for the container:
  - Create `text-to-speech-v1/.env` with: `HF_TOKEN=your_token_here`
  - In `text-to-speech-v1/docker-compose.yml`, add under each service:
    ```
    environment:
      - HF_TOKEN
    ```
  - Rebuild/restart via the batch scripts.

8) Where to find results

- Audio files: `text-to-speech-v1/assets/outputs/<timestamp>/<subfolder>/<lang>/...`
- Processed texts: `text-to-speech-v1/assets/inputs/processed/<timestamp>/<subfolders>/name_<timestamp>.bak`

## Chatterbox

- This CLI uses the Multilingual TTS model from the Chatterbox repo: https://github.com/resemble-ai/chatterbox
- The Docker image installs Chatterbox from GitHub and CPU PyTorch wheels. On first run, the model will be downloaded.
- Optional: provide a reference audio file to guide voice/style.

## Assets

- Inputs:  text-to-speech-v1/assets/inputs
- Outputs: text-to-speech-v1/assets/outputs

Outputs are ignored by git by default (see .gitignore). The folders include a `.gitkeep` so the structure stays in the repo.

## Build and Push

1) Build:
   docker build -t <your-registry>/<your-repo>/tts-cli:latest text-to-speech-v1

2) Login to your registry (example for Docker Hub):
   docker login

3) Push:
   docker push <your-registry>/<your-repo>/tts-cli:latest

Rename the image/tag as you prefer.

## Local (optional)

If you want to run without Docker and you can use pip:

- Create a venv and install deps:
  python -m venv .venv
  .venv\\Scripts\\activate  # Windows
  # source .venv/bin/activate # macOS/Linux
  # CPU only
  pip install --index-url https://download.pytorch.org/whl/cpu torch
  # or GPU (CUDA 12.x) â€“ example for CUDA 12.4 matching chatterbox torch==2.6.0
  pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
  pip install -r text-to-speech-v1/requirements.txt

- Run the CLI:
  python text-to-speech-v1/cli.py

Note: On Linux, `espeak` and `ffmpeg` are recommended for best results:
  sudo apt-get install espeak ffmpeg

## Usage

- Interactive (prompts):
  python text-to-speech-v1/cli.py

- Non-interactive (flags):
  python text-to-speech-v1/cli.py \
    --input-dir text-to-speech-v1/assets/inputs \
    --output-dir text-to-speech-v1/assets/outputs \
    --formats wav,ogg \
    --language-id en \
    --temperature 0.8 --exaggeration 0.5 --cfg-weight 0.5 \
    --seed 0 --device auto \
    --processed-dir-name processed \
    --overwrite --non-interactive

### Windows Quick Start (Batch)

1. Run `text-to-speech-v1/setup.bat` (installs deps; installs CUDA wheels if GPU present).
2. Place `.txt` files in `text-to-speech-v1/assets/inputs/`.
3. Choose a batch:
   - `start_auto_tts.bat`: default behavior â€” timestamped run folder, auto language, single WAV per text (chunks concatenated).
   - `start_auto_tts_split_chunks.bat`: writes separate files per chunk for long texts.
   - `start_auto_tts_no_timestamp.bat`: writes directly under `assets/outputs` (still split by language folders), no timestamp subfolder.

Outputs are written under `assets/outputs/<timestamp>/<lang>/...` (unless using the no-timestamp variant).

Processed inputs move to `assets/inputs/processed/<timestamp>/...` and are renamed to `.bak` with the timestamp appended (e.g., `hello_2025-09-20_133418.bak`). When `--recursive` is used, the original subfolder structure is preserved under the processed folder.

### Windows Quick Start (Batch)

1. Run `text-to-speech-v1/setup.bat` (installs deps, installs CUDA wheels if GPU present).
2. Place `.txt` files in `text-to-speech-v1/assets/inputs/`.
3. Run `text-to-speech-v1/start_auto_tts.bat`.

Outputs are written to `assets/outputs/<timestamp>/<lang>/...` and inputs are moved to `assets/inputs/processed/`.

### Language detection and output structure

- Auto language detection is on by default; unrecognized texts fall back to `--language-id`.
- Outputs are grouped under a timestamped folder and language code, e.g.:
  `assets/outputs/2025-09-20_133418/en/hello.wav`.

### Long text behavior

- Texts of any length are supported. The CLI chunks long text by sentence and concatenates into a single WAV per input by default.
- To keep per-chunk files instead, pass `--split-chunks`.

### Audio controls and examples

- temperature: Controls sampling randomness. Higher = more variety, lower = more deterministic. Try 0.6â€“0.9. Example: `--temperature 0.7` for steadier speech.
- exaggeration: Style/emotion intensity for delivery (0â€“1). Higher = more expressive. Example: `--exaggeration 0.7` to make it livelier.
- cfg-weight: Classifier-free guidance strength (0â€“1). Higher = stronger adherence to conditioning but can reduce naturalness if too high. Typical 0.3â€“0.7. Example: `--cfg-weight 0.4` for a good balance.
- seed: Random seed for reproducibility. `0` means random each run; any non-zero value repeats behavior. Example: `--seed 42` to make runs repeatable.

Examples:
- Conversational, expressive: `--temperature 0.85 --exaggeration 0.8 --cfg-weight 0.45`
- Neutral, steady: `--temperature 0.6 --exaggeration 0.3 --cfg-weight 0.5 --seed 1234`

### Silence trimming

- The CLI trims leading/trailing silence by default using an energy-based detector. This helps avoid long tails of silence, especially in some languages like Chinese.
- Adjust sensitivity with `--silence-top-db` (default 40). Larger values trim more aggressively; smaller values retain more pauses.
- Disable with `--no-trim-silence`.

### Advanced flags

- `--processed-dir-name <name>`: subfolder under inputs for moved files.
- `--no-auto-language`: disable detection (force `--language-id`).
- `--no-timestamp-dir`: write outputs directly under `--output-dir` (no timestamp subfolder).
- `--timestamp-format <fmt>`: strftime for timestamp folder. In Windows batch files, escape `%` as `%%`.
- `--recursive`: include subfolders when scanning for input files.

## Notes

- Language must be one of the supported ids printed by the CLI.
- If `ffmpeg` is not available, only WAV will be produced.
- Empty files are skipped.
- First run will download the model; subsequent runs reuse it.
## Repository layout

- `src/tts_cli/cli.py` – core CLI implementation used by scripts and Docker images.
- `cli.py` – compatibility shim that forwards to the packaged CLI (adds `src/` to `PYTHONPATH`).
- `tests/` – unit tests covering every helper; run them with `python run_tests.py`.
- `examples/tts_config.toml` - default config used by the optional single-text example.
- `examples/tts_from_text.py` – optional config-driven helper kept as a sample outside the main CLI.

## Example Output
- 女士们，先生们，晚上好。我是今天的司仪，Bryan。
非常荣幸在此与各位相识，一同见证如此神圣的一刻。
证婚典礼将在 5–10 分钟内举行。
- [01_pre-announcement.wav](https://github.com/user-attachments/files/22452582/01_pre-announcement.wav)
- Good evening, ladies and gentlemen. I am Bryan, your emcee today.
It’s a pleasure to see all of you here to witness this wonderful event.
Please be informed that the wedding solemnization will begin in about 5–10 minutes.
- [01_pre-announcement.wav](https://github.com/user-attachments/files/22452588/01_pre-announcement.wav)

