from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


MODEL_IDS = {
    "1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}


def _inject_windows_truststore() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


def _pick_runtime():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn_implementation = "sdpa" if device == "cuda" else None
    return device, dtype, attn_implementation


def _load_model(model_key: str):
    from qwen_tts import Qwen3TTSModel

    if model_key not in MODEL_IDS:
        raise ValueError(f"Unknown Qwen3-TTS model key: {model_key}")
    device, dtype, attn_implementation = _pick_runtime()
    return Qwen3TTSModel.from_pretrained(
        MODEL_IDS[model_key],
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def _first_audio(wavs) -> np.ndarray:
    if isinstance(wavs, (list, tuple)):
        return np.asarray(wavs[0])
    return np.asarray(wavs)


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> tuple[int, np.ndarray]:
    if not target_sr or int(source_sr) == int(target_sr):
        return int(source_sr), audio

    import librosa

    resampled = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=int(source_sr), target_sr=int(target_sr))
    return int(target_sr), np.asarray(resampled)


def _write_wav(path: Path, audio: np.ndarray, sr: int, target_sr: int = 0) -> None:
    final_sr, final_audio = _resample_audio(audio, int(sr), int(target_sr or 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), final_audio, final_sr, subtype="PCM_16")


def _read_text_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        clean = line.strip()
        if clean:
            lines.append(clean)
    return lines


def generate(args: argparse.Namespace) -> tuple[int, np.ndarray]:
    if args.mode == "design":
        if not args.text:
            raise ValueError("--text is required for design mode.")
        model = _load_model("1.7B-VoiceDesign")
        wavs, sr = model.generate_voice_design(
            text=args.text,
            language=args.language,
            instruct=args.instruct or "",
        )
        return int(sr), _first_audio(wavs)

    if args.mode == "clone":
        if not args.text:
            raise ValueError("--text is required for clone mode.")
        if not args.ref_wav:
            raise ValueError("--ref-wav is required for clone mode.")
        if not args.ref_text:
            raise ValueError("--ref-text is required for Qwen3-TTS clone mode.")
        model = _load_model(args.clone_model)
        prompt = model.create_voice_clone_prompt(
            ref_audio=args.ref_wav,
            ref_text=args.ref_text,
            x_vector_only_mode=False,
        )
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            voice_clone_prompt=prompt,
        )
        return int(sr), _first_audio(wavs)

    raise ValueError(f"Unknown mode: {args.mode}")


def _write_progress(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_clone_batch(args: argparse.Namespace) -> tuple[Path, Path, int, int]:
    if not args.ref_wav:
        raise ValueError("--ref-wav is required for clone-batch mode.")
    if not args.ref_text:
        raise ValueError("--ref-text is required for clone-batch mode.")
    if not args.texts_file:
        raise ValueError("--texts-file is required for clone-batch mode.")
    if not args.output_dir:
        raise ValueError("--output-dir is required for clone-batch mode.")

    texts = _read_text_lines(Path(args.texts_file))
    if not texts:
        raise ValueError("--texts-file does not contain any non-empty lines.")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    text_list_path = output_dir / (args.text_list or "Neutral.txt")
    progress_path = output_dir / "_progress.json"
    failed_path = output_dir / "_failed.jsonl"
    retry_path = output_dir / "_retry_texts.txt"
    raw_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.clone_model)
    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_wav,
        ref_text=args.ref_text,
        x_vector_only_mode=False,
    )

    text_list_rows = []
    failed_rows = []
    retry_texts = []
    _write_progress(
        progress_path,
        {"status": "running", "total": len(texts), "done": 0, "succeeded": 0, "failed": 0, "current": ""},
    )
    for index, text in enumerate(texts, start=1):
        stem = f"{index:04d}"
        wav_path = raw_dir / f"{stem}.wav"
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=args.language,
                voice_clone_prompt=prompt,
            )
            _write_wav(wav_path, _first_audio(wavs), int(sr), int(args.target_sr or 0))
            text_list_rows.append(f"{stem}|{text}")
            print(f"[ok {index}/{len(texts)}] {wav_path.name}", flush=True)
        except Exception as exc:
            failed_rows.append({"index": index, "stem": stem, "text": text, "error": str(exc)})
            retry_texts.append(text)
            print(f"[failed {index}/{len(texts)}] {stem}: {exc}", flush=True)
        finally:
            _write_progress(
                progress_path,
                {
                    "status": "running",
                    "total": len(texts),
                    "done": index,
                    "succeeded": len(text_list_rows),
                    "failed": len(failed_rows),
                    "current": stem,
                },
            )

    text_list_path.write_text("\n".join(text_list_rows) + "\n", encoding="utf-8")
    if failed_rows:
        failed_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in failed_rows) + "\n",
            encoding="utf-8",
        )
        retry_path.write_text("\n".join(retry_texts) + "\n", encoding="utf-8")
    else:
        for path in (failed_path, retry_path):
            if path.exists():
                path.unlink()
    _write_progress(
        progress_path,
        {
            "status": "complete" if text_list_rows else "failed",
            "total": len(texts),
            "done": len(texts),
            "succeeded": len(text_list_rows),
            "failed": len(failed_rows),
            "current": "",
        },
    )
    if not text_list_rows:
        raise RuntimeError(f"All {len(texts)} corpus rows failed. See {failed_path}")
    return raw_dir, text_list_path, len(text_list_rows), len(failed_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin Qwen3-TTS inference wrapper for JP Voice Studio.")
    parser.add_argument("--mode", choices=["design", "clone", "clone-batch"], required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--output-wav", default="")
    parser.add_argument("--texts-file", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--text-list", default="Neutral.txt")
    parser.add_argument("--target-sr", type=int, default=0)
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--ref-wav", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--clone-model", choices=["1.7B-Base", "0.6B-Base"], default="1.7B-Base")
    args = parser.parse_args()

    _inject_windows_truststore()
    if args.mode == "clone-batch":
        raw_dir, text_list_path, count, failed = generate_clone_batch(args)
        print(f"Generated {count} wav files: {raw_dir}")
        if failed:
            print(f"Failed rows: {failed}")
        print(f"Text list: {text_list_path}")
        return

    if not args.output_wav:
        raise ValueError("--output-wav is required for design and clone modes.")
    output_path = Path(args.output_wav)
    sr, audio = generate(args)
    _write_wav(output_path, audio, sr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Qwen3-TTS inference failed: {exc}", file=sys.stderr)
        raise
