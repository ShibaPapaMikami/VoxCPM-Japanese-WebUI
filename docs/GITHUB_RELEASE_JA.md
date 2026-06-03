# GitHub公開チェックリスト

## 公開してよいもの

- ソースコード
- `scripts/*.ps1`
- `VoxCPM_WebUI.cmd`
- `README_SETUP_JA.md`
- `LICENSE`
- 変更内容を説明するREADME

## 公開しないもの

- `pretrained_models/`
- `outputs/`
- `external/`
- `.venv/`
- `.uv-cache/`
- Hugging Face / ModelScope のキャッシュ
- 生成音声
- 社内音声、参照音声、文字起こしデータ
- 個人PC固有の設定ファイル
- `VoxCPM_WebUI.exe`

## ライセンス表記

OpenBMB/VoxCPM と VoxCPM2 は Apache-2.0 ライセンスです。Irodori-TTS、Irodori-TTS-500M-v3、Semantic-DACVAE-Japanese-32dim はMITライセンスです。Qwen3-TTS と qwen-tts は Apache-2.0 ライセンスです。Voice-Design-Cloner は MIT ライセンスです。

公開リポジトリでは `LICENSE` と [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) を残し、READMEに以下のような説明を入れてください。

```text
JP Voice Studio is an unofficial Japanese Web UI and integration layer based on OpenBMB/VoxCPM, with optional Irodori-TTS and simplified VoiceDesignCloner/Qwen3-TTS workflow support.
Original project: https://github.com/OpenBMB/VoxCPM
OpenBMB/VoxCPM and VoxCPM2: Apache-2.0
Irodori-TTS and related optional Irodori models/codecs: MIT
Qwen3-TTS and qwen-tts: Apache-2.0
Voice-Design-Cloner: MIT
```

## READMEに入れる推奨注意書き

```text
Voice cloning should only be used with voices you own or have explicit permission to use.
Do not impersonate real people or publish generated audio in a misleading way.
Users are responsible for complying with applicable laws, contracts, and platform policies.
```

## 公開前コマンド

巨大ファイルや生成物が入っていないか確認します。

```powershell
git status --short
git ls-files
```

Gitに載る予定のファイルに100MB以上のものがないか確認します。

```powershell
git ls-files |
  ForEach-Object { Get-Item $_ } |
  Where-Object { $_.Length -gt 100MB } |
  Select-Object FullName,Length
```

GitHubに上げる前に、`pretrained_models/` と `outputs/` が表示されないことを確認してください。

## 社内配布のおすすめ

エンジニア向け:

```powershell
git clone <repo-url>
cd <repo>
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1
```

2回目以降:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1 -SkipBaseSetup
```

Irodori-TTSやQwen3-TTSも使う場合:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1 -AllEngines
```

非エンジニア向けには、GitHub Releasesにzipを置き、READMEに上記コマンドだけを案内するのが扱いやすいです。
