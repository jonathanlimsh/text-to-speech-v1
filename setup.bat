@echo off
call venv\Scripts\activate
cmd
python -m pip install --upgrade pip
pip install "numpy<1.26"
pip install wheel
pip install -r requirements.txt

REM If an NVIDIA GPU is present, install CUDA-enabled PyTorch wheels (matching chatterbox torch==2.6.0)
where nvidia-smi >nul 2>&1 \
  && (
    echo Detected NVIDIA GPU. Installing CUDA-enabled PyTorch 2.6.0 wheels...
    python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
  ) \
  || (
    echo No NVIDIA GPU detected via nvidia-smi. Skipping CUDA PyTorch install.
  )
if not exist examples\tts_config.toml copy examples\tts_config.example.toml examples\tts_config.toml

REM Prompt user to run start_auto_tts.bat
set /p runAutoTTS=Do you want to auto-generate speech now? (y/n): 
if /i "%runAutoTTS%"=="y" (
    call start_auto_tts.bat
) else (
    echo Skipping auto-generation of speech.
)
