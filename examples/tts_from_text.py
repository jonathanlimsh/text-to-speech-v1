import argparse
import os
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# Prefer stdlib tomllib (Py>=3.11); fall back to tomli if present
try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # Will error if used on <3.11


def _ensure_src_on_path():
    """Ensure local src/ is importable for `chatterbox` package when running from repo."""
    here = Path(__file__).resolve().parent
    src = here / "src"
    if src.exists():
        sys.path.insert(0, str(src))


_ensure_src_on_path()

import torch  # noqa: E402
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES  # noqa: E402


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if tomllib is None:
        raise RuntimeError("tomllib not available. Use Python 3.11+ or install tomli.")
    with path.open("rb") as f:
        return tomllib.load(f)


def read_text_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    # Model is tuned for shorter prompts; follow app’s 300-char cap
    return text[:300]


def pick_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref in {"cuda", "cpu"}:
        return pref
    return "cpu"


def save_wav_int16(waveform: np.ndarray, sample_rate: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Expect float waveform in [-1, 1]; convert to int16 PCM
    wav = np.asarray(waveform, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)

    # Ensure mono 1-D
    if pcm.ndim > 1:
        pcm = pcm.squeeze()

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def synth_from_config(cfg: dict) -> Path:
    input_text_file = Path(cfg.get("input_text_file", "")).expanduser()
    output_path = Path(cfg.get("output_path", "output.wav")).expanduser()
    language_id = str(cfg.get("language_id", "en")).lower()
    audio_prompt_path: Optional[str] = cfg.get("audio_prompt_path") or None
    exaggeration = float(cfg.get("exaggeration", 0.5))
    temperature = float(cfg.get("temperature", 0.8))
    cfg_weight = float(cfg.get("cfg_weight", 0.5))
    seed = int(cfg.get("seed", 0))
    device_pref = str(cfg.get("device", "auto")).lower()

    if language_id not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        raise ValueError(f"Unsupported language_id '{language_id}'. Supported: {supported}")

    if not input_text_file.exists():
        raise FileNotFoundError(f"Input text file not found: {input_text_file}")

    text = read_text_file(input_text_file)
    device = pick_device(device_pref)

    print(f"Using device: {device}")
    print(f"Language: {language_id} ({SUPPORTED_LANGUAGES[language_id]})")
    if audio_prompt_path:
        print(f"Reference audio: {audio_prompt_path}")
    else:
        print("Reference audio: <none> (using default voice)")

    # Seed for reproducibility
    if seed != 0:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Load model and synthesize
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    wav = model.generate(
        text,
        language_id=language_id,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    # model.generate returns a torch tensor [1, T]; convert to numpy 1-D
    wav_np = wav.squeeze(0).detach().cpu().numpy()
    save_wav_int16(wav_np, model.sr, output_path)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize speech from a text file using Chatterbox Multilingual TTS (config-driven)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("tts_config.toml"),
        help="Path to TOML config file (default: examples/tts_config.toml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    synth_from_config(cfg)


if __name__ == "__main__":
    main()

