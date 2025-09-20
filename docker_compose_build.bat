@echo off
setlocal

REM Change to repo folder of this script
pushd "%~dp0" >nul 2>&1

REM Verify Docker is available
where docker >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker is not installed or not in PATH.
  echo         Install Docker Desktop, then re-run this script.
  popd & exit /b 1
)

REM Detect compose command (v2 plugin or legacy v1)
set "DC=docker compose"
%DC% version >nul 2>&1
if errorlevel 1 (
  set "DC=docker-compose"
  %DC% version >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Docker Compose not found. Install Docker Desktop with Compose v2 or docker-compose.
    popd & exit /b 1
  )
)

REM Locate compose file (avoid FOR to handle parentheses in paths)
set "COMPOSE_FILE="
if exist "docker-compose.yml" set "COMPOSE_FILE=docker-compose.yml"
if not defined COMPOSE_FILE if exist "compose.yml" set "COMPOSE_FILE=compose.yml"
if not defined COMPOSE_FILE if exist "docker-compose.yaml" set "COMPOSE_FILE=docker-compose.yaml"
if not defined COMPOSE_FILE if exist "compose.yaml" set "COMPOSE_FILE=compose.yaml"

if not defined COMPOSE_FILE (
  echo [ERROR] No compose file found in current directory.
  echo         Expected one of: docker-compose.yml, compose.yml, docker-compose.yaml, compose.yaml
  popd & exit /b 1
)

echo Using compose file: %COMPOSE_FILE%

REM Optional profile argument: cpu | gpu | all (default: all)
set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=all"

set "PROFILE_FLAGS="
if /I "%PROFILE%"=="cpu" set "PROFILE_FLAGS=--profile cpu"
if /I "%PROFILE%"=="gpu" set "PROFILE_FLAGS=--profile gpu"
if /I "%PROFILE%"=="all" set "PROFILE_FLAGS=--profile cpu --profile gpu"

echo Building images with profiles: %PROFILE%
%DC% -f "%COMPOSE_FILE%" %PROFILE_FLAGS% build --pull
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
  echo [ERROR] Build failed with code %ERR%.
  popd & exit /b %ERR%
)

echo [OK] Build completed.
popd & exit /b 0
