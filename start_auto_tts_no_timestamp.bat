@echo off
call venv\Scripts\activate
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

python -m tts_cli.cli ^
  --input-dir assets/inputs ^
  --output-dir assets/outputs ^
  --formats wav,ogg ^
  --language-id en ^
  --processed-dir-name processed ^
  --temperature 0.8 --exaggeration 0.5 --cfg-weight 0.5 ^
  --seed 0 --device auto ^
  --no-timestamp-dir ^
  --overwrite --non-interactive

deactivate

