from __future__ import annotations

import argparse
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


def generate(args: argparse.Namespace) -> tuple[int, np.ndarray]:
    if args.mode == "design":
        model = _load_model("1.7B-VoiceDesign")
        wavs, sr = model.generate_voice_design(
            text=args.text,
            language=args.language,
            instruct=args.instruct or "",
        )
        return int(sr), _first_audio(wavs)

    if args.mode == "clone":
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin Qwen3-TTS inference wrapper for JP Voice Studio.")
    parser.add_argument("--mode", choices=["design", "clone"], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--ref-wav", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--clone-model", choices=["1.7B-Base", "0.6B-Base"], default="1.7B-Base")
    args = parser.parse_args()

    _inject_windows_truststore()
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sr, audio = generate(args)
    sf.write(str(output_path), audio, sr, subtype="PCM_16")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Qwen3-TTS inference failed: {exc}", file=sys.stderr)
        raise
