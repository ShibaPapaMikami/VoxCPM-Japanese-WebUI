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
- `.venv/`
- `.uv-cache/`
- Hugging Face / ModelScope のキャッシュ
- 生成音声
- 社内音声、参照音声、文字起こしデータ
- 個人PC固有の設定ファイル
- `VoxCPM_WebUI.exe`

## ライセンス表記

OpenBMB/VoxCPM と VoxCPM2 は Apache-2.0 ライセンスです。公開リポジトリでは `LICENSE` を残し、READMEに以下のような説明を入れてください。

```text
This project is based on OpenBMB/VoxCPM and includes local UI/Windows setup modifications.
Original project: https://github.com/OpenBMB/VoxCPM
License: Apache-2.0
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
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1
```

2回目以降:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1 -SkipSetup
```

非エンジニア向けには、GitHub Releasesにzipを置き、READMEに上記コマンドだけを案内するのが扱いやすいです。
