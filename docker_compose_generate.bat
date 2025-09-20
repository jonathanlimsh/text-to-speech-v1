@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Change to repo folder of this script
pushd "%~dp0" >nul 2>&1

REM Load .env (same file Compose uses). Lines like KEY=VALUE, ignore comments.
if exist ".env" (
  for /f "usebackq tokens=1* delims==" %%A in (".env") do (
    set "k=%%A"
    set "v=%%B"
    if not "!k!"=="" if not "!k:~0,1!"=="#" set "!k!=!v!"
  )
)

REM Auto-detect timezone offset (+/-HHMM) unless provided
if "!TZ_OFFSET!"=="" (
  for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "(Get-Date).ToString('zzz').Replace(':','')"`) do set "TZ_OFFSET=%%T"
)
set "DOCKER_ENV_ARGS="
if not "!TZ_OFFSET!"=="" set "DOCKER_ENV_ARGS=--env TZ_OFFSET=!TZ_OFFSET!"

REM Verify Docker
where docker >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker not found in PATH.
  popd & exit /b 1
)

REM Detect compose command
set "DC=docker compose"
%DC% version >nul 2>&1
if errorlevel 1 (
  set "DC=docker-compose"
  %DC% version >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Docker Compose not found.
    popd & exit /b 1
  )
)

REM Locate compose file
set "COMPOSE_FILE="
if exist "docker-compose.yml" set "COMPOSE_FILE=docker-compose.yml"
if not defined COMPOSE_FILE if exist "compose.yml" set "COMPOSE_FILE=compose.yml"
if not defined COMPOSE_FILE if exist "docker-compose.yaml" set "COMPOSE_FILE=docker-compose.yaml"
if not defined COMPOSE_FILE if exist "compose.yaml" set "COMPOSE_FILE=compose.yaml"
if not defined COMPOSE_FILE (
  echo [ERROR] No compose file found in current directory.
  popd & exit /b 1
)

REM Decide service from env or first arg (auto -> pick GPU if available)
set "SERVICE=!SERVICE!"
if not "%~1"=="" set "SERVICE=%~1"
if "!SERVICE!"=="" set "SERVICE=auto"
if /I "!SERVICE!"=="auto" (
  where nvidia-smi >nul 2>&1 && ( set "SERVICE=tts-gpu" ) || ( set "SERVICE=tts-cpu" )
) else (
  if /I "!SERVICE!"=="gpu" set "SERVICE=tts-gpu"
  if /I "!SERVICE!"=="cpu" set "SERVICE=tts-cpu"
)
echo Using service: !SERVICE!

REM Shift off service arg
if not "%~1"=="" shift

REM Resolve input dir: prefer INPUT_DIR from env, else INPUT_SUBDIR under /app/assets/inputs
set "INPUT_DIR=!INPUT_DIR!"
if "!INPUT_DIR!"=="" (
  set "INPUT_DIR=/app/assets/inputs"
  if not "!INPUT_SUBDIR!"=="" set "INPUT_DIR=/app/assets/inputs/!INPUT_SUBDIR!"
)

REM Optional next arg overrides input (subfolder or absolute / path)
set "CAND=%~1"
if not "%CAND%"=="" (
  echo %CAND% | findstr /b /c:"--" >nul
  if errorlevel 1 (
    if "%CAND:~0,1%"=="/" (
      set "INPUT_DIR=%CAND%"
    ) else (
      set "INPUT_DIR=/app/assets/inputs/%CAND%"
    )
    shift
  )
)

REM Defaults
if "!FORMATS!"=="" set "FORMATS=wav"
if "!LANGUAGE_ID!"=="" set "LANGUAGE_ID=en"
if "!DEVICE!"=="" set "DEVICE=auto"

REM Flags from booleans
set "REC=" & if /I "!RECURSIVE!"=="true" set "REC=--recursive"
set "NOTS=" & if /I "!NO_TIMESTAMP!"=="true" set "NOTS=--no-timestamp-dir"
set "SCH=" & if /I "!SPLIT_CHUNKS!"=="true" set "SCH=--split-chunks"
set "OVR=--overwrite" & if /I "!OVERWRITE!"=="false" set "OVR="
set "NTS=" & if /I "!TRIM_SILENCE!"=="false" set "NTS=--no-trim-silence"
set "TTO=" & if /I "!TRIM_TAIL_ONLY!"=="true" set "TTO=--trim-tail-only"
set "PDN=" & if not "!PROCESSED_DIR_NAME!"=="" set "PDN=--processed-dir-name !PROCESSED_DIR_NAME!"
set "STD=" & if not "!SILENCE_TOP_DB!"=="" set "STD=--silence-top-db !SILENCE_TOP_DB!"
set "SMD=" & if not "!SILENCE_MIN_DUR!"=="" set "SMD=--silence-min-dur !SILENCE_MIN_DUR!"
set "SPM=" & if not "!SILENCE_PAD_MS!"=="" set "SPM=--silence-pad-ms !SILENCE_PAD_MS!"

REM Run generation (ENTRYPOINT already runs cli.py)
%DC% -f "%COMPOSE_FILE%" run --rm !SERVICE! !DOCKER_ENV_ARGS! ^
  --input-dir "!INPUT_DIR!" ^
  --output-dir /app/assets/outputs ^
  --formats "!FORMATS!" ^
  --language-id "!LANGUAGE_ID!" ^
  --device "!DEVICE!" ^
  !REC! !NOTS! !SCH! !OVR! !NTS! !TTO! !PDN! !STD! !SMD! !SPM! ^
  --non-interactive
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
  echo [ERROR] Generation failed with code %ERR%.
  popd & exit /b %ERR%
)

echo [OK] Generation completed.
popd & exit /b 0
