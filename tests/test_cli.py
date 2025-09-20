import argparse
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Provide lightweight stubs for optional heavy dependencies.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def manual_seed(*_args, **_kwargs):
            return None

        @staticmethod
        def manual_seed_all(*_args, **_kwargs):
            return None

    torch_stub.cuda = types.SimpleNamespace(
        is_available=_Cuda.is_available,
        manual_seed=_Cuda.manual_seed,
        manual_seed_all=_Cuda.manual_seed_all,
    )

    def _manual_seed(*_args, **_kwargs):  # pragma: no cover - stubbed helper
        return None

    torch_stub.manual_seed = _manual_seed
    sys.modules["torch"] = torch_stub

if "chatterbox" not in sys.modules:
    sys.modules["chatterbox"] = types.ModuleType("chatterbox")

if "chatterbox.mtl_tts" not in sys.modules:
    mtl_tts_module = types.ModuleType("chatterbox.mtl_tts")

    class DummyTTS:
        sr = 24000

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):  # pragma: no cover - patched in tests
            raise NotImplementedError("Stubbed out")

    mtl_tts_module.ChatterboxMultilingualTTS = DummyTTS
    mtl_tts_module.SUPPORTED_LANGUAGES = {"en": "English"}
    sys.modules["chatterbox.mtl_tts"] = mtl_tts_module
    sys.modules["chatterbox"].mtl_tts = mtl_tts_module

from tts_cli import cli


class FakeTensor:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)

    def squeeze(self, axis=None):
        return FakeTensor(np.squeeze(self._data, axis=axis))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._data, dtype=np.float32)


class FormatTimestampTests(unittest.TestCase):
    def test_default_format_includes_timezone(self):
        stamp = cli.format_timestamp()
        self.assertRegex(stamp, r"^\d{4}-\d{2}-\d{2}_\d{6}[+-]\d{4}$")

    def test_custom_format_hhmm_timezone(self):
        stamp = cli.format_timestamp("%H%M%z")
        self.assertRegex(stamp, r"^\d{4}[+-]\d{4}$")

    def test_invalid_format_falls_back(self):
        stamp = cli.format_timestamp("%Q")
        self.assertRegex(stamp, r"^\d{4}-\d{2}-\d{2}_\d{6}[+-]\d{4}$")

    def test_format_timestamp_respects_env_offset(self):
        with mock.patch.dict(os.environ, {"TZ_OFFSET": "+0530"}, clear=False):
            stamp = cli.format_timestamp("%z")
        self.assertEqual(stamp, "+0530")


class PromptConfigInteractiveTests(unittest.TestCase):
    def test_prompt_config_interactive_accepts_defaults(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        inputs = base / "inputs"
        inputs.mkdir()
        outputs = base / "outputs"

        answers = iter([
            str(inputs),
            "",
            str(outputs),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ])

        with mock.patch("builtins.input", side_effect=lambda _=None: next(answers)):
            with mock.patch.dict(cli.SUPPORTED_LANGUAGES, {"en": "English", "es": "Spanish"}, clear=True):
                cfg = cli.prompt_config_interactive(argparse.Namespace())

        self.assertEqual(cfg.input_dir, inputs.resolve())
        self.assertEqual(cfg.output_dir, outputs.resolve())
        self.assertEqual(cfg.formats, ["wav"])
        self.assertTrue(cfg.auto_language)
        self.assertEqual(cfg.language_id, "en")
        self.assertTrue(cfg.use_timestamp_dir)
        self.assertTrue(cfg.trim_silence)


class ReadTextFileTests(unittest.TestCase):
    def test_read_text_file_strips_whitespace(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "sample.txt"
        path.write_text(" hello world \n", encoding="utf-8")
        self.assertEqual(cli.read_text_file(path), "hello world")

    def test_read_text_file_handles_decode_errors(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "latin.txt"
        path.write_bytes("H?llo".encode("cp1252"))
        self.assertEqual(cli.read_text_file(path), "H?llo")


class TrimSilenceTests(unittest.TestCase):
    def test_tail_trim_uses_librosa_path(self):
        wav = np.concatenate([np.ones(2000, dtype=np.float32), np.zeros(1000, dtype=np.float32)])

        fake_librosa = types.SimpleNamespace(
            feature=types.SimpleNamespace(
                rms=lambda **_kwargs: np.array([[1.0, 0.0]], dtype=np.float32)
            ),
            effects=types.SimpleNamespace(
                split=lambda y, **_kwargs: np.array([[0, len(y)]], dtype=np.int64)
            ),
        )

        with mock.patch.dict(sys.modules, {"librosa": fake_librosa}):
            trimmed = cli.trim_trailing_and_leading_silence(
                wav, sr=48000, min_silence_dur=0.0, pad_ms=0
            )

        self.assertLess(len(trimmed), len(wav))
        self.assertEqual(len(trimmed), 480)

    def test_trim_falls_back_without_librosa(self):
        wav = np.concatenate([np.ones(64, dtype=np.float32), np.zeros(32, dtype=np.float32)])

        def _boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        crashy = types.SimpleNamespace(
            feature=types.SimpleNamespace(rms=_boom),
            effects=types.SimpleNamespace(split=_boom),
        )

        with mock.patch.dict(sys.modules, {"librosa": crashy}):
            trimmed = cli.trim_trailing_and_leading_silence(wav, sr=16000, pad_ms=0)

        self.assertLess(len(trimmed), len(wav))
        self.assertGreater(len(trimmed), 0)


class TextChunkingTests(unittest.TestCase):
    def test_sentences_from_text(self):
        text = "Hello world! How are you? This is fine."
        sentences = cli.sentences_from_text(text)
        self.assertEqual(sentences, ["Hello world!", "How are you?", "This is fine."])

    def test_dynamic_chunks_long_text(self):
        text = "a" * 2500
        chunks = list(cli.dynamic_chunks(text))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)


class LanguageDetectionTests(unittest.TestCase):
    def test_detect_language_supported(self):
        with mock.patch("tts_cli.cli.detect_lang", return_value="es"):
            with mock.patch.dict(cli.SUPPORTED_LANGUAGES, {"en": "English", "es": "Spanish"}, clear=True):
                self.assertEqual(cli.detect_language_code("hola"), "es")

    def test_detect_language_fallback_on_error(self):
        with mock.patch("tts_cli.cli.detect_lang", side_effect=Exception("oops")):
            result = cli.detect_language_code("text", fallback="en")
        self.assertEqual(result, "en")


class DeviceSelectionTests(unittest.TestCase):
    def test_pick_device_auto_prefers_cuda(self):
        with mock.patch.object(cli.torch.cuda, "is_available", return_value=True):
            self.assertEqual(cli.pick_device("auto"), "cuda")

    def test_pick_device_invalid_defaults_to_cpu(self):
        self.assertEqual(cli.pick_device("invalid"), "cpu")


class SaveWavTests(unittest.TestCase):
    def test_save_wav_int16_writes_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "test.wav"
        cli.save_wav_int16(np.array([0.1, -0.1], dtype=np.float32), 16000, path)
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)


class FfmpegTests(unittest.TestCase):
    def test_ensure_ffmpeg_true(self):
        with mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            self.assertTrue(cli.ensure_ffmpeg())

    def test_convert_with_ffmpeg_mp3(self):
        wav_path = Path("sample.wav")
        with mock.patch("tts_cli.cli.subprocess.check_call") as mocked_call:
            dst = cli.convert_with_ffmpeg(wav_path, "mp3")
        mocked_call.assert_called_once()
        self.assertEqual(dst, wav_path.with_suffix(".mp3"))

    def test_convert_with_ffmpeg_wav_passthrough(self):
        wav_path = Path("sample.wav")
        with mock.patch("tts_cli.cli.subprocess.check_call") as mocked_call:
            dst = cli.convert_with_ffmpeg(wav_path, "wav")
        mocked_call.assert_not_called()
        self.assertEqual(dst, wav_path)

    def test_convert_with_ffmpeg_invalid(self):
        with self.assertRaises(ValueError):
            cli.convert_with_ffmpeg(Path("sample.wav"), "flac")


class GenerateFromFolderTests(unittest.TestCase):
    def test_generate_from_folder_basic_flow(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        input_dir = base / "in"
        output_dir = base / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        text_file = input_dir / "hello.txt"
        text_file.write_text("Hello there", encoding="utf-8")

        config = cli.Config(input_dir=input_dir, output_dir=output_dir)
        config.formats = ["wav"]
        config.auto_language = False
        config.use_timestamp_dir = False
        config.trim_silence = False
        config.preserve_subdirs = False
        config.overwrite = True

        class FakeModel:
            sr = 16000

            def generate(self, *_args, **_kwargs):
                return FakeTensor([[0.0, 0.1, -0.1]])

        saved_paths = []

        def fake_save(_data, _sr, out_path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("data", encoding="utf-8")
            saved_paths.append(out_path)

        with mock.patch.object(cli.ChatterboxMultilingualTTS, "from_pretrained", return_value=FakeModel()):
            with mock.patch.object(cli, "save_wav_int16", side_effect=fake_save):
                cli.generate_from_folder(config)

        self.assertEqual(len(saved_paths), 1)
        expected = output_dir / "en" / "hello.wav"
        self.assertEqual(saved_paths[0], expected)
        processed_path = (
            input_dir
            / config.processed_dir_name
            / output_dir.name
            / f"{text_file.stem}_{output_dir.name}.bak"
        )
        self.assertFalse(text_file.exists())
        self.assertTrue(processed_path.exists())


class ParseArgsTests(unittest.TestCase):
    def test_parse_args(self):
        args = cli.parse_args([
            "--input-dir", "inputs",
            "--output-dir", "outputs",
            "--formats", "wav,mp3",
            "--language-id", "en",
            "--non-interactive",
        ])
        self.assertEqual(args.input_dir, "inputs")
        self.assertEqual(args.output_dir, "outputs")
        self.assertEqual(args.formats, "wav,mp3")
        self.assertTrue(args.non_interactive)


class MainFunctionTests(unittest.TestCase):
    def test_main_quits_when_user_declines(self):
        with mock.patch("builtins.input", return_value="n"):
            with mock.patch("tts_cli.cli.parse_args") as parse_mock:
                result = cli.main([])
        self.assertEqual(result, 0)
        parse_mock.assert_not_called()

    def test_main_runs_non_interactive(self):
        ns = argparse.Namespace(
            input_dir="inputs",
            output_dir="outputs",
            pattern="*.txt",
            formats="wav",
            language_id="en",
            auto_language=True,
            audio_prompt_path=None,
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.5,
            seed=0,
            device="auto",
            overwrite=True,
            processed_dir_name="processed",
            use_timestamp_dir=True,
            timestamp_format=cli.DEFAULT_TIMESTAMP_FORMAT,
            split_chunks=False,
            recursive=False,
            trim_silence=True,
            silence_top_db=40.0,
            trim_tail_only=True,
            silence_min_dur=0.3,
            silence_pad_ms=30,
            preserve_subdirs=True,
            non_interactive=True,
        )

        with mock.patch("tts_cli.cli.parse_args", return_value=ns):
            with mock.patch("tts_cli.cli.generate_from_folder") as gen_mock:
                result = cli.main(["--non-interactive"])

        self.assertEqual(result, 0)
        gen_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
