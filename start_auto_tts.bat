@echo off
setlocal ENABLEDELAYEDEXPANSION
REM Ensure we run from this script's directory
pushd "%~dp0" >nul 2>&1

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

call venv\Scripts\activate

python -m tts_cli.cli ^
  --input-dir "D:\Jonathan\Life\3 Career Storyline\(2025-09-08) Self Study\2 Projects and Exercises\Text To Speech\text-to-speech-v1\assets\inputs\Wedding_Emcee_Scripts_EN_ZH_Sequenced\Emcee_Scripts_Chinese" ^
  --output-dir assets/outputs ^
  --formats wav ^
  --processed-dir-name processed ^
  --temperature 0.8 --exaggeration 0.5 --cfg-weight 0.5 ^
  --seed 0 --device auto ^
  --recursive ^
  --overwrite --non-interactive

deactivate
popd >nul 2>&1
