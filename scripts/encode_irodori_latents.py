from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from irodori_tts.codec import DACVAECodec  # noqa: E402


TARGET_SR = 48000
NORMALIZE_DB = -16.0
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


def _print_err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _loudness_normalize(wav: torch.Tensor, target_db: float) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
    current_db = 20 * torch.log10(rms)
    gain = 10 ** ((target_db - current_db) / 20)
    return (wav * gain).clamp(-1.0, 1.0)


def _load_wav(path: str, target_sr: int) -> torch.Tensor:
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        return wav.float()
    except Exception as torchaudio_exc:
        try:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                import torchaudio

                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
            return wav.float()
        except Exception as sf_exc:
            raise RuntimeError(
                f"both torchaudio and soundfile failed to load {path}: "
                f"torchaudio={torchaudio_exc} soundfile={sf_exc}"
            ) from sf_exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode Irodori-TTS LoRA training WAVs to DACVAE latents.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    latent_dir = Path(args.latent_dir)
    manifest_path = Path(args.manifest)
    latent_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    _print_err(f"[encode] {len(entries)} entries")
    _print_err(f"[encode] codec: {CODEC_REPO}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = DACVAECodec.load(
        repo_id=CODEC_REPO,
        device=device,
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=None,
    )

    written = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as out_f:
        for index, entry in enumerate(entries):
            wav_path = entry["audio"]
            text = entry["text"]
            try:
                wav = _load_wav(wav_path, TARGET_SR)
                wav = _loudness_normalize(wav, NORMALIZE_DB)
                with torch.inference_mode():
                    latent = codec.encode_waveform(wav, sample_rate=TARGET_SR)[0].cpu()
            except Exception as exc:
                _print_err(f"[skip] {wav_path}: {exc}")
                skipped += 1
                continue

            latent_path = latent_dir / f"{written:08d}.pt"
            torch.save(latent, latent_path)
            rel_latent = os.path.relpath(latent_path, start=manifest_path.parent).replace(os.sep, "/")
            out_f.write(
                json.dumps(
                    {"text": text, "latent_path": rel_latent, "num_frames": int(latent.shape[0])},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if (index + 1) % 25 == 0 or index == len(entries) - 1:
                _print_err(f"[encode] {index + 1}/{len(entries)}")

    _print_err(f"[done] written={written} skipped={skipped}")
    if written == 0:
        _print_err("[error] no entries were encoded. Check the [skip] lines above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
