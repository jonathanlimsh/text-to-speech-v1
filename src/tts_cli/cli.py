import argparse
import os
import re
import sys
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Iterable

import numpy as np

import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

try:
    # Lightweight language detection
    from langdetect import detect as detect_lang
except Exception:
    detect_lang = None


DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M%S%z"


TZ_OFFSET_PATTERN = re.compile(r"^[+-]\d{4}$")


def _parse_env_timezone():
    raw = os.getenv("TZ_OFFSET", "").strip()
    if raw and TZ_OFFSET_PATTERN.match(raw):
        hours = int(raw[1:3])
        minutes = int(raw[3:5])
        offset = timedelta(hours=hours, minutes=minutes)
        if raw[0] == "-":
            offset = -offset
        try:
            return timezone(offset)
        except Exception:
            return None
    return None

def format_timestamp(fmt: str = DEFAULT_TIMESTAMP_FORMAT) -> str:
    """Format current time using optional TZ override and fall back to default format on errors."""
    tz = _parse_env_timezone()
    if tz is not None:
        now = datetime.now(tz)
    else:
        now = datetime.now().astimezone()
    try:
        return now.strftime(fmt)
    except Exception:
        return now.strftime(DEFAULT_TIMESTAMP_FORMAT)

@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    pattern: str = "*.txt"
    formats: List[str] = None  # e.g., ["wav", "mp3", "ogg"]
    language_id: str = "en"  # fallback / default
    auto_language: bool = True
    audio_prompt_path: Optional[Path] = None
    exaggeration: float = 0.5
    temperature: float = 0.8
    cfg_weight: float = 0.5
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda
    overwrite: bool = False
    processed_dir_name: str = "processed"
    use_timestamp_dir: bool = True
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT
    combine_chunks: bool = True
    recursive: bool = False
    trim_silence: bool = True
    silence_top_db: float = 40.0
    trim_tail_only: bool = True
    silence_min_dur: float = 0.3  # seconds of continuous silence to trim
    silence_pad_ms: int = 30      # pad this much after trimming
    preserve_subdirs: bool = True


def prompt_config_interactive(args: argparse.Namespace) -> Config:
    print("Configure TTS generation for a folder (Chatterbox).")

    def prompt_path(prompt: str, default: Optional[str] = None, must_exist: bool = False) -> Path:
        while True:
            raw = input(f"{prompt} [{default or ''}]: ").strip()
            val = raw or (default or "")
            p = Path(val).expanduser().resolve() if val else None
            if not p:
                print("Please provide a valid path.")
                continue
            if must_exist and not p.exists():
                print(f"Path does not exist: {p}")
                continue
            return p

    default_input = Path("text-to-speech-v1/assets/inputs").resolve()
    default_output = Path("text-to-speech-v1/assets/outputs").resolve()
    input_dir = prompt_path("Input folder containing text files", str(default_input), must_exist=True)
    pattern = input("File pattern (e.g., *.txt) [*.txt]: ").strip() or "*.txt"
    output_dir = prompt_path("Output folder for audio files", str(default_output), must_exist=False)

    fmt_default = "wav"
    formats_raw = input("Output formats (csv: wav,mp3,ogg) [wav]: ").strip() or fmt_default
    formats = [f.strip().lower() for f in formats_raw.split(',') if f.strip()]
    for f in formats:
        if f not in {"wav", "mp3", "ogg"}:
            print(f"Unsupported format '{f}'. Supported: wav, mp3, ogg")
            sys.exit(2)

    # Language selection / detection
    langs = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
    auto_language = True
    auto_q = input("Auto-detect language? (Y/n) [Y]: ").strip().lower()
    if auto_q == "n":
        auto_language = False
    language_id = "en"
    if not auto_language:
        language_id = input(f"Language id [{langs}] [en]: ").strip() or "en"
        if language_id not in SUPPORTED_LANGUAGES:
            print(f"Invalid language_id '{language_id}'. Falling back to 'en'.")
            language_id = "en"

    # Optional reference audio for voice cloning/style
    audio_prompt_raw = input("Reference audio path (optional) []: ").strip() or ""
    audio_prompt_path = Path(audio_prompt_raw).expanduser().resolve() if audio_prompt_raw else None
    if audio_prompt_path and not audio_prompt_path.exists():
        print(f"Warning: reference audio not found: {audio_prompt_path}")
        audio_prompt_path = None

    def prompt_float(prompt: str, default: float, lo: float, hi: float) -> float:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print(f"Please enter a number between {lo} and {hi}.")
        return prompt_float(prompt, default, lo, hi)

    def prompt_int(prompt: str, default: int) -> int:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            return prompt_int(prompt, default)

    exaggeration = prompt_float("Exaggeration (0-1)", 0.5, 0.0, 1.0)
    temperature = prompt_float("Temperature (0-1)", 0.8, 0.0, 1.0)
    cfg_weight = prompt_float("CFG weight (0-1)", 0.5, 0.0, 1.0)
    seed = prompt_int("Seed (0 for random)", 0)
    device = input("Device [auto|cpu|cuda] [auto]: ").strip().lower() or "auto"
    if device not in {"auto", "cpu", "cuda"}:
        print("Invalid device, using 'auto'.")
        device = "auto"

    overwrite = (input("Overwrite existing files? (y/N) [N]: ").strip().lower() == "y")

    # Processed dir name
    processed_dir_name = input("Processed subfolder name [processed]: ").strip() or "processed"

    # Timestamped run folder
    use_timestamp_dir = True
    ts_q = input("Create timestamped run folder? (Y/n) [Y]: ").strip().lower()
    if ts_q == "n":
        use_timestamp_dir = False
    timestamp_format = DEFAULT_TIMESTAMP_FORMAT
    if use_timestamp_dir:
        ts_fmt_in = input(f"Timestamp format for folder [{DEFAULT_TIMESTAMP_FORMAT}]: ").strip()
        if ts_fmt_in:
            timestamp_format = ts_fmt_in

    # Recursive processing
    rec_q = input("Process subfolders recursively? (Y/n) [Y]: ").strip().lower()
    recursive = (rec_q != "n")

    # Silence trimming
    trim_q = input("Trim silence from start/end? (Y/n) [Y]: ").strip().lower()
    trim_silence = (trim_q != "n")
    silence_top_db = 40.0
    if trim_silence:
        silence_top_db = prompt_float("Silence trim top_db (10-80)", 40.0, 10.0, 80.0)

    cfg = Config(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
        formats=formats,
        language_id=language_id,
        auto_language=auto_language,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfg_weight,
        seed=seed,
        device=device,
        overwrite=overwrite,
        processed_dir_name=processed_dir_name,
        use_timestamp_dir=use_timestamp_dir,
        timestamp_format=timestamp_format,
        combine_chunks=True,
        recursive=recursive,
        trim_silence=trim_silence,
        silence_top_db=silence_top_db,
    )

    print("\nSummary:")
    print(f" - Input dir:   {cfg.input_dir}")
    print(f" - Pattern:     {cfg.pattern}")
    print(f" - Output dir:  {cfg.output_dir}")
    print(f" - Formats:     {', '.join(cfg.formats)}")
    print(f" - Language:    {cfg.language_id} ({SUPPORTED_LANGUAGES.get(cfg.language_id, '?')})")
    print(f" - Auto-lang:   {cfg.auto_language}")
    print(f" - Ref audio:   {cfg.audio_prompt_path or '[none]'}")
    print(f" - Exaggeration:{cfg.exaggeration}")
    print(f" - Temperature: {cfg.temperature}")
    print(f" - CFG weight:  {cfg.cfg_weight}")
    print(f" - Seed:        {cfg.seed}")
    print(f" - Device:      {cfg.device}")
    print(f" - Run folder:  {'timestamped' if cfg.use_timestamp_dir else 'output root'} ({cfg.timestamp_format if cfg.use_timestamp_dir else '-'})")
    print(f" - Processed:   {cfg.processed_dir_name}")
    print(f" - Recursive:   {cfg.recursive}")
    print(f" - TrimSilence: {cfg.trim_silence} (top_db={cfg.silence_top_db})")
    ok = input("Proceed? (Y/n) [Y]: ").strip().lower()
    if ok == "n":
        print("Aborted by user.")
        sys.exit(1)
    return cfg


def read_text_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(errors="ignore")
    return text.strip()


def trim_trailing_and_leading_silence(
    waveform: np.ndarray,
    sr: int,
    top_db: float = 40.0,
    tail_only: bool = True,
    min_silence_dur: float = 0.3,
    pad_ms: int = 30,
) -> np.ndarray:
    try:
        import librosa
        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.squeeze()
        if tail_only:
            # Tail-focused trim using windowed RMS to resist tiny noises
            frame = max(256, int(0.02 * sr))  # ~20ms
            hop = max(128, int(0.01 * sr))    # ~10ms
            rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop, center=False).flatten()
            if rms.size == 0:
                return wav
            # Threshold relative to max RMS
            thr = float(np.max(rms)) * (10.0 ** (-abs(top_db) / 20.0))
            below = rms < thr
            # Require continuous below-threshold for min_silence_dur seconds
            need = max(1, int(min_silence_dur / (hop / sr)))
            run = 0
            cut_frame = None
            for i in range(rms.size - 1, -1, -1):
                run = run + 1 if below[i] else 0
                if run >= need:
                    cut_frame = i
                    break
            if cut_frame is None:
                return wav
            end_sample = min(len(wav), (cut_frame * hop) - int((pad_ms / 1000.0) * sr))
            end_sample = max(0, end_sample)
            return wav[:end_sample]
        else:
            intervals = librosa.effects.split(wav, top_db=float(top_db), frame_length=2048, hop_length=512)
            if intervals.size == 0:
                return wav
            start = int(intervals[0, 0])
            end = int(intervals[-1, 1])
            # apply padding
            pad = int((pad_ms / 1000.0) * sr)
            start = max(0, start - pad)
            end = min(len(wav), end + pad)
            return wav[start:end]
    except Exception:
        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.squeeze()
        # Simple amplitude-based fallback with tail-only trim
        idx = np.where(np.abs(wav) > 2e-3)[0]
        if idx.size == 0:
            return wav
        if tail_only:
            pad = int((pad_ms / 1000.0) * sr)
            return wav[: max(idx[-1] - pad, 0)]
        return wav[idx[0]: idx[-1] + 1]


def sentences_from_text(text: str) -> List[str]:
    # Split on common sentence terminators while keeping them attached
    # Supports latin and CJK punctuation and Arabic question mark
    pattern = r".+?(?:[\.\!\?。！？？]|$)"
    sents = [s.strip() for s in re.findall(pattern, text, flags=re.S) if s and s.strip()]
    return sents or [text]


def dynamic_chunks(text: str) -> Iterable[str]:
    # Determine a dynamic target chunk size (~ up to 1000 chars)
    L = len(text)
    if L <= 1200:
        yield text
        return
    # Aim for ~1000 characters per chunk, adjust based on total length
    n = max(2, math.ceil(L / 1000))
    target = max(600, min(1200, math.ceil(L / n)))
    hard_max = 1400

    # Split text into sentences using ASCII and common Unicode enders
    _pat = ".+?(?:[\\.\\!\\?]|\\u3002|\\uFF01|\\u061F|$)"
    sents = [s.strip() for s in re.findall(_pat, text, flags=re.S) if s and s.strip()]
    buf = []
    cur = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) > hard_max:
            # Long sentence: yield in windows
            if cur:
                yield " ".join(buf).strip()
                buf, cur = [], 0
            for i in range(0, len(s), target):
                yield s[i:i+target]
            continue
        if cur + len(s) + (1 if cur else 0) <= target:
            buf.append(s)
            cur += len(s) + (1 if cur else 0)
        else:
            if buf:
                yield " ".join(buf).strip()
            buf = [s]
            cur = len(s)
    if buf:
        yield " ".join(buf).strip()


def detect_language_code(text: str, fallback: str = "en") -> str:
    # Map detector output to supported language ids
    if detect_lang is None:
        return fallback
    try:
        code = detect_lang(text)
    except Exception:
        return fallback
    code = (code or "").lower()
    # Normalize variants
    if code in {"zh-cn", "zh-tw", "zh-hans", "zh-hant", "zh_trad", "zh_simp"}:
        code = "zh"
    if code == "iw":  # legacy Hebrew code
        code = "he"
    # If unsupported, fallback
    return code if code in SUPPORTED_LANGUAGES else fallback


def pick_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref in {"cuda", "cpu"}:
        return pref
    return "cpu"


def save_wav_int16(waveform: np.ndarray, sample_rate: int, out_path: Path) -> None:
    import wave

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(waveform, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)
    if pcm.ndim > 1:
        pcm = pcm.squeeze()
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def ensure_ffmpeg() -> bool:
    from shutil import which

    return which("ffmpeg") is not None


def convert_with_ffmpeg(wav_path: Path, out_format: str) -> Path:
    dst = wav_path.with_suffix(f".{out_format}")
    if out_format == "mp3":
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            str(dst),
        ]
    elif out_format == "ogg":
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav_path),
            "-codec:a", "libvorbis",
            "-q:a", "5",
            str(dst),
        ]
    elif out_format == "wav":
        return wav_path
    else:
        raise ValueError(f"Unsupported format: {out_format}")
    subprocess.check_call(cmd)
    return dst


def generate_from_folder(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Resolve run directory (timestamped or root)
    if cfg.use_timestamp_dir:
        run_tag = format_timestamp(cfg.timestamp_format)
        run_dir = (cfg.output_dir / run_tag).resolve()
    else:
        run_dir = cfg.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.auto_language and cfg.language_id not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        raise ValueError(f"Unsupported language_id '{cfg.language_id}'. Supported: {supported}")
    files = sorted(cfg.input_dir.rglob(cfg.pattern)) if cfg.recursive else sorted(cfg.input_dir.glob(cfg.pattern))
    if not files:
        print(f"No files matched pattern '{cfg.pattern}' in {cfg.input_dir}")
        return

    print(f"Found {len(files)} file(s). Using Chatterbox TTS...")
    print(f"Run directory: {run_dir}")
    need_ffmpeg = any(fmt in {"mp3", "ogg"} for fmt in cfg.formats)
    if need_ffmpeg and not ensure_ffmpeg():
        print("ffmpeg not found. Only WAV will be produced.")
        cfg.formats = [f for f in cfg.formats if f == "wav"] or ["wav"]

    # Seed for reproducibility
    if cfg.seed != 0:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed(cfg.seed)
                torch.cuda.manual_seed_all(cfg.seed)
            except Exception:
                pass
        try:
            np.random.seed(cfg.seed)
        except Exception:
            pass

    device = pick_device(cfg.device)
    print(f"Using device: {device}")
    print(f"Language: {cfg.language_id} ({SUPPORTED_LANGUAGES.get(cfg.language_id, '?')})")
    if cfg.audio_prompt_path:
        print(f"Reference audio: {cfg.audio_prompt_path}")
    else:
        print("Reference audio: <none> (using default voice)")

    # Load model once
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    for src in files:
        text = read_text_file(src)
        if not text:
            print(f"Skipping empty file: {src.name}")
            continue
        base = src.stem
        # Determine language for this file
        lang = cfg.language_id
        if cfg.auto_language:
            lang = detect_language_code(text, fallback=cfg.language_id)
        lang_name = SUPPORTED_LANGUAGES.get(lang, "?")
        if lang not in SUPPORTED_LANGUAGES:
            print(f"Detected unsupported language for {src.name}: '{lang}', falling back to '{cfg.language_id}'")
            lang = cfg.language_id
            lang_name = SUPPORTED_LANGUAGES.get(lang, "?")
        # Choose output directory: preserve subfolders if requested
        if cfg.preserve_subdirs:
            try:
                rel_dir = src.parent.relative_to(cfg.input_dir)
            except Exception:
                rel_dir = Path("")
            out_base_dir = run_dir / rel_dir / lang
        else:
            out_base_dir = run_dir / lang
        lang_dir = out_base_dir
        lang_dir.mkdir(parents=True, exist_ok=True)

        chunks = list(dynamic_chunks(text))
        many = len(chunks) > 1

        try:
            out_paths: List[Path] = []
            if cfg.combine_chunks and many:
                # Synthesize all chunks and concatenate into a single file
                wavs = []
                for idx, chunk in enumerate(chunks, start=1):
                    wav_t = model.generate(
                        chunk,
                        language_id=lang,
                        audio_prompt_path=str(cfg.audio_prompt_path) if cfg.audio_prompt_path else None,
                        exaggeration=cfg.exaggeration,
                        cfg_weight=cfg.cfg_weight,
                        temperature=cfg.temperature,
                    )
                    wav_np = wav_t.squeeze(0).detach().cpu().numpy()
                    wavs.append(wav_np)
                full = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
                if cfg.trim_silence:
                    full = trim_trailing_and_leading_silence(full, model.sr, cfg.silence_top_db)
                final_wav = lang_dir / f"{base}.wav"
                if final_wav.exists() and not cfg.overwrite:
                    print(f"Exists, skipping synthesis: {final_wav.relative_to(cfg.output_dir)}")
                else:
                    # Trim silence if enabled
                    if cfg.trim_silence:
                        full = trim_trailing_and_leading_silence(
                            full, model.sr,
                            top_db=cfg.silence_top_db,
                            tail_only=cfg.trim_tail_only,
                            min_silence_dur=cfg.silence_min_dur,
                            pad_ms=cfg.silence_pad_ms,
                        )
                    save_wav_int16(full, model.sr, final_wav)
                    print(f"Wrote: {final_wav.relative_to(cfg.output_dir)} [{lang} {lang_name}] (combined {len(chunks)} chunks)")
                out_paths.append(final_wav)
            else:
                # Write each chunk as a separate file
                for idx, chunk in enumerate(chunks, start=1):
                    stem = f"{base}_{idx:03d}" if many else base
                    tmp_wav = lang_dir / f"{stem}.wav"
                    if tmp_wav.exists() and not cfg.overwrite:
                        print(f"Exists, skipping synthesis: {tmp_wav.relative_to(cfg.output_dir)}")
                    else:
                        wav = model.generate(
                            chunk,
                            language_id=lang,
                            audio_prompt_path=str(cfg.audio_prompt_path) if cfg.audio_prompt_path else None,
                            exaggeration=cfg.exaggeration,
                            cfg_weight=cfg.cfg_weight,
                            temperature=cfg.temperature,
                        )
                        wav_np = wav.squeeze(0).detach().cpu().numpy()
                        if cfg.trim_silence:
                            wav_np = trim_trailing_and_leading_silence(
                                wav_np, model.sr,
                                top_db=cfg.silence_top_db,
                                tail_only=cfg.trim_tail_only,
                                min_silence_dur=cfg.silence_min_dur,
                                pad_ms=cfg.silence_pad_ms,
                            )
                        save_wav_int16(wav_np, model.sr, tmp_wav)
                        print(f"Wrote: {tmp_wav.relative_to(cfg.output_dir)} [{lang} {lang_name}]")
                    out_paths.append(tmp_wav)

            # Convert to requested formats for resulting files
            for tmp_wav in out_paths:
                for fmt in cfg.formats:
                    if fmt == "wav":
                        continue
                    out_path = tmp_wav.with_suffix(f".{fmt}")
                    if out_path.exists() and not cfg.overwrite:
                        print(f"Exists, skipping: {out_path.relative_to(cfg.output_dir)}")
                        continue
                    try:
                        convert_with_ffmpeg(tmp_wav, fmt)
                        print(f"Wrote: {out_path.relative_to(cfg.output_dir)}")
                    except Exception as e:
                        print(f"Error converting to {fmt} for {src.name}: {e}")
        except Exception as e:
            print(f"Error synthesizing {src.name}: {e}")
            continue

        # Move processed file to input_dir/processed/<run_tag>/... and rename to .bak with timestamp
        try:
            processed_root = cfg.input_dir / cfg.processed_dir_name / run_dir.name
            rel = src.relative_to(cfg.input_dir)
            dst_dir = processed_root / rel.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{src.stem}_{run_dir.name}.bak"
            if dst.exists() and not cfg.overwrite:
                # add numeric suffix
                for k in range(1, 1000):
                    candidate = dst_dir / f"{src.stem}_{run_dir.name}_{k}.bak"
                    if not candidate.exists():
                        dst = candidate
                        break
            shutil.move(str(src), str(dst))
        except Exception as e:
            print(f"Warning: failed to move processed file {src.name}: {e}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate audio from a folder of text files using Chatterbox TTS.")
    p.add_argument("--input-dir", type=str, help="Folder containing text files")
    p.add_argument("--output-dir", type=str, help="Folder for generated audio files")
    p.add_argument("--pattern", type=str, default="*.txt", help="Glob pattern for input files (default: *.txt)")
    p.add_argument("--formats", type=str, default="wav", help="Comma-separated output formats (wav,mp3,ogg)")
    p.add_argument("--language-id", type=str, default="en", help="Fallback language id, e.g. en, es, fr")
    p.add_argument("--no-auto-language", dest="auto_language", action="store_false", help="Disable auto language detection")
    p.set_defaults(auto_language=True)
    p.add_argument("--audio-prompt-path", type=str, default=None, help="Optional reference audio file path")
    p.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration 0-1")
    p.add_argument("--temperature", type=float, default=0.8, help="Temperature 0-1")
    p.add_argument("--cfg-weight", type=float, default=0.5, help="Classifier-free guidance weight 0-1")
    p.add_argument("--seed", type=int, default=0, help="Seed for reproducibility (0 = random)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--processed-dir-name", type=str, default="processed", help="Subfolder under input dir to move processed files")
    p.add_argument("--timestamp-format", type=str, default=DEFAULT_TIMESTAMP_FORMAT, help="strftime format for output run folder name")
    p.add_argument("--no-timestamp-dir", dest="use_timestamp_dir", action="store_false", help="Write outputs directly under output dir (no timestamp subfolder)")
    p.set_defaults(use_timestamp_dir=True)
    p.add_argument("--split-chunks", action="store_true", help="Do not combine long text chunks; write multiple files per input")
    p.add_argument("--recursive", action="store_true", help="Process subfolders recursively")
    p.add_argument("--no-trim-silence", dest="trim_silence", action="store_false", help="Disable trimming leading/trailing silence in outputs")
    p.set_defaults(trim_silence=True)
    p.add_argument("--silence-top-db", type=float, default=40.0, help="Sensitivity for silence trimming (higher trims more)")
    p.add_argument("--trim-tail-only", action="store_true", help="Only trim trailing silence (default)")
    p.add_argument("--silence-min-dur", type=float, default=0.3, help="Seconds of continuous silence required before trimming tail")
    p.add_argument("--silence-pad-ms", type=int, default=30, help="Padding after trim to avoid abrupt cut (ms)")
    p.add_argument("--no-preserve-subdirs", dest="preserve_subdirs", action="store_false", help="Do not mirror input subfolders under the run directory")
    p.set_defaults(preserve_subdirs=True, trim_tail_only=True)
    p.add_argument("--non-interactive", action="store_true", help="Do not prompt; require flags")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    # Prompt at the very start
    if not any(arg in argv for arg in ["--non-interactive", "-h", "--help"]):
        proceed = input("Start text-to-speech generation? (y/n): ").strip().lower()
        if proceed != "y":
            print("Quitting.")
            return 0
        
    args = parse_args(argv)
    # If non-interactive or all required args passed, build config from args
    if args.non_interactive or (args.input_dir and args.output_dir):
        formats = [f.strip().lower() for f in (args.formats or "wav").split(',') if f.strip()]
        cfg = Config(
            input_dir=Path(args.input_dir).expanduser().resolve() if args.input_dir else Path.cwd(),
            output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else Path.cwd(),
            pattern=args.pattern,
            formats=formats,
            language_id=(args.language_id or "en").lower(),
            auto_language=bool(args.auto_language),
            audio_prompt_path=Path(args.audio_prompt_path).expanduser().resolve() if args.audio_prompt_path else None,
            exaggeration=float(args.exaggeration),
            temperature=float(args.temperature),
            cfg_weight=float(args.cfg_weight),
            seed=int(args.seed),
            device=args.device or "auto",
            overwrite=args.overwrite,
            processed_dir_name=str(args.processed_dir_name or "processed"),
            use_timestamp_dir=bool(args.use_timestamp_dir),
            timestamp_format=str(args.timestamp_format or DEFAULT_TIMESTAMP_FORMAT),
            combine_chunks=not bool(args.split_chunks),
            recursive=bool(args.recursive),
            trim_silence=bool(args.trim_silence),
            silence_top_db=float(args.silence_top_db),
            trim_tail_only=bool(args.trim_tail_only),
            silence_min_dur=float(args.silence_min_dur),
            silence_pad_ms=int(args.silence_pad_ms),
            preserve_subdirs=bool(args.preserve_subdirs),
        )
    else:
        cfg = prompt_config_interactive(args)

    try:
        generate_from_folder(cfg)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
