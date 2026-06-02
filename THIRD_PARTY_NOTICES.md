# Third-Party Notices

This project, JP Voice Studio, is an unofficial Japanese Web UI and integration layer for open-source speech generation engines. It includes local UI, Windows setup, engine switching, and workflow improvements.

このファイルは、GitHub公開・社内配布時に確認しやすいよう、主な第三者プロジェクト、モデル、ライセンス、利用上の注意をまとめたものです。最終的な利用条件は、各公式リポジトリおよびモデル配布ページの最新表記を確認してください。

## Included Or Referenced Projects

| Component | Purpose | Source | License |
| --- | --- | --- | --- |
| OpenBMB/VoxCPM | Base TTS implementation and VoxCPM2 support | https://github.com/OpenBMB/VoxCPM | Apache-2.0 |
| VoxCPM2 model | Multilingual TTS, voice design, voice cloning | https://huggingface.co/openbmb/VoxCPM2 | Apache-2.0 |
| Aratako/Irodori-TTS | Optional Japanese-specialized TTS engine | https://github.com/Aratako/Irodori-TTS | MIT |
| Irodori-TTS-500M-v3 | Optional Irodori model checkpoint | https://huggingface.co/Aratako/Irodori-TTS-500M-v3 | MIT |
| Semantic-DACVAE-Japanese-32dim | Codec used by Irodori-TTS | https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim | MIT |
| Qwen3-TTS models | Optional multilingual voice design and voice cloning engine | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | Apache-2.0 |
| qwen-tts | Optional Python package used to run Qwen3-TTS | https://pypi.org/project/qwen-tts/ | Apache-2.0 |
| reinehonoka/Voice-Design-Cloner | Reference implementation for Qwen3-TTS voice design / clone workflows | https://github.com/reinehonoka/Voice-Design-Cloner | MIT |
| Gradio | Local Web UI framework | https://github.com/gradio-app/gradio | Apache-2.0 |

## License Summary

The licenses above generally allow use, modification, redistribution, and commercial use, provided their conditions are followed.

主な条件:

- Keep the applicable copyright and license notices.
- Do not remove or misrepresent upstream attribution.
- Do not imply that this repository is an official OpenBMB or Aratako project.
- Review the latest upstream license and model card before production or external distribution.

## Repository Distribution Policy

The following files and directories should not be committed or published:

- `pretrained_models/`
- `outputs/`
- `external/`
- `.venv/`
- `.uv-cache/`
- Hugging Face / ModelScope caches
- generated audio files
- reference voice recordings
- transcripts containing personal, internal, or confidential data
- local settings such as `.jpvoxcpm_settings.json`
- packaged local executables such as `VoxCPM_WebUI.exe`

The current `.gitignore` is configured to exclude the major generated and local-only folders.

## Voice And Safety Notice

Voice cloning and speech generation can affect privacy, publicity rights, copyright, labor contracts, and platform policies. Users are responsible for lawful and ethical use.

Use this project only with:

- your own voice,
- voices for which you have explicit permission,
- public-domain or properly licensed material,
- internal test voices that do not identify a real person without consent.

Do not use this project to:

- impersonate a real person without clear consent,
- create misleading statements attributed to someone else,
- commit fraud, harassment, or deception,
- bypass identity, biometric, or voice-authentication systems,
- publish generated audio in a way that hides that it was AI-generated when disclosure is required.

## Commercial Use

Based on the currently referenced upstream licenses, this project can generally be used commercially and published on GitHub, as long as the license notices and safety constraints above are respected.

For company or client deployment, also check:

- internal legal / compliance requirements,
- consent records for reference voices,
- dataset and input-audio provenance,
- whether generated audio needs AI disclosure,
- local laws in the countries where the tool and audio are used.

## Attribution Text

Recommended attribution for README, release notes, or internal documentation:

```text
JP Voice Studio is an unofficial Japanese Web UI and integration layer based on OpenBMB/VoxCPM, with optional Irodori-TTS support.

OpenBMB/VoxCPM and VoxCPM2 are licensed under Apache-2.0.
Irodori-TTS, Irodori-TTS-500M-v3, and Semantic-DACVAE-Japanese-32dim are licensed under MIT.
Qwen3-TTS and qwen-tts are licensed under Apache-2.0.
Voice-Design-Cloner is licensed under MIT and is referenced for Qwen3-TTS workflow integration.
Users are responsible for using generated or cloned voices only with appropriate permission and in compliance with applicable laws and policies.
```
