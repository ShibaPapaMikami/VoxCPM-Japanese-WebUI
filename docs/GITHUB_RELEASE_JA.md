# GitHub Releases 配布手順

JP Voice StudioをGitHub Releasesで配布するときのチェックリストです。モデル、生成音声、社内音声、外部エンジン本体をReleaseに混ぜないことを最優先にします。

## 配布方針

推奨する配布形態:

- GitHubリポジトリ: ソースコード、起動スクリプト、ドキュメントを公開する
- GitHub Releases: バージョンごとの説明、導入手順、必要に応じてソースzipを添付する
- モデル本体: 初回セットアップ時に各公式配布元から取得する
- 生成音声、参照音声、社内データ: 配布しない

GitHubが自動生成する `Source code (zip)` だけでも、基本的には配布できます。独自zipを添付する場合は、この手順で中身を確認してください。

## 公開してよいもの

- ソースコード
- `scripts/*.ps1`
- `VoxCPM_WebUI.cmd`
- `JPVoiceStudio_Setup.cmd`
- `README.md`
- `README_SETUP_JA.md`
- `docs/*.md`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- アプリで使うロゴや静的画像

## 公開しないもの

- `pretrained_models/`
- `outputs/`
- `external/`
- `.venv/`
- `.uv-cache/`
- `uv-cache-*/`
- Hugging Face / ModelScope のキャッシュ
- 生成音声
- 社内音声、参照音声、文字起こしデータ
- 個人PC固有の設定ファイル
- `.env`
- `.jpvoxcpm_settings.json`
- `VoxCPM_WebUI.exe`

`VoxCPM_WebUI.exe` はローカルで作った実行ファイルなので、Releaseに含める場合は [Windows実行ファイルの作り直し手順](./WINDOWS_LAUNCHER_BUILD_JA.md) に従って作り直してください。

## ライセンス表記

OpenBMB/VoxCPM と VoxCPM2 は Apache-2.0 ライセンスです。Irodori-TTS、Irodori-TTS-500M-v3、Semantic-DACVAE-Japanese-32dim はMITライセンスです。Qwen3-TTS と qwen-tts は Apache-2.0 ライセンスです。Voice-Design-Cloner は MIT ライセンスです。

公開リポジトリでは `LICENSE` と [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) を残し、READMEに以下の趣旨が入っていることを確認します。

```text
JP Voice Studio is an unofficial Japanese Web UI and integration layer based on OpenBMB/VoxCPM, with optional Irodori-TTS and simplified VoiceDesignCloner/Qwen3-TTS workflow support.
Original project: https://github.com/OpenBMB/VoxCPM
OpenBMB/VoxCPM and VoxCPM2: Apache-2.0
Irodori-TTS and related optional Irodori models/codecs: MIT
Qwen3-TTS and qwen-tts: Apache-2.0
Voice-Design-Cloner: MIT
```

## 利用上の注意

READMEまたはRelease本文に、以下の注意を入れてください。

```text
声のクローン機能は、自分の声、または本人から明示的な許可を得た音声だけに使用してください。
生成音声を実在人物の発言として偽装したり、第三者の権利を侵害する用途には使わないでください。
利用者は、適用される法律、契約、社内規程、プラットフォームポリシーを守る責任があります。
```

## Release前チェック

作業前に最新状態を確認します。

```powershell
git status --short
git log --oneline -5
```

未追跡ファイルがある場合は、Releaseに必要なものか確認します。候補画像、生成音声、検証用ファイルは原則として含めません。

巨大ファイルや生成物がGit管理対象に入っていないか確認します。

```powershell
git ls-files |
  ForEach-Object { Get-Item $_ } |
  Where-Object { $_.Length -gt 100MB } |
  Select-Object FullName,Length
```

以下が表示されないことを確認します。

- `pretrained_models/`
- `outputs/`
- `external/`
- `.venv/`
- `.env`
- 個人名や社内名を含む音声・テキスト

## 最小動作確認

Release前に最低限、次を確認します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_setup.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_smoke_jp.ps1 `
  -Text "今日はリリース前の動作確認をしています。" `
  -Control "calm Japanese male narration" `
  -Output "outputs\release_smoke_voxcpm2.wav" `
  -InferenceTimesteps 4
```

Web UIも確認します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\launch_webui.ps1 -Port 8808
```

ブラウザで `http://127.0.0.1:8808/` を開き、[最小スモークテスト手順](./SMOKE_TEST_JA.md) に沿って確認します。

## 独自zipを作る場合

GitHubの自動生成zipではなく、添付用zipを作りたい場合だけ実行します。

```powershell
$Version = "v0.1.0"
$Dist = "dist"
$Zip = "$Dist\JP-Voice-Studio-$Version-source.zip"
New-Item -ItemType Directory -Force $Dist | Out-Null
git archive --format=zip --output=$Zip HEAD
Get-Item $Zip | Select-Object FullName,Length
```

zipの中身を確認します。

```powershell
tar -tf $Zip | Select-String "pretrained_models|outputs|external|.venv|.env|VoxCPM_WebUI.exe"
```

何も表示されなければ、配布対象外ファイルは混ざっていません。

## GitHub Release作成手順

1. GitHubのリポジトリ画面を開きます。
2. `Releases` を開きます。
3. `Draft a new release` を押します。
4. タグを作成します。例: `v0.1.0`
5. Release titleを書きます。例: `JP Voice Studio v0.1.0`
6. 下のRelease本文テンプレートを貼ります。
7. 独自zipを作った場合だけ添付します。
8. `VoxCPM_WebUI.exe` を添付する場合は、作り直し手順に沿ってビルド・検証してから添付します。
9. `Publish release` を押します。

## Release本文テンプレート

````markdown
## JP Voice Studio v0.1.0

日本語で使いやすい音声生成・声クローン統合Web UIです。

### 主な内容

- VoxCPM2による声のデザイン、声のクローン、高精度クローン
- Irodori-TTS任意連携
- VoiceDesignCloner/Qwen3-TTS簡易連携
- 日本語UI、WAVダウンロード、声の履歴、録音ガイド

### セットアップ

```powershell
git clone https://github.com/ShibaPapaMikami/VoxCPM-Japanese-WebUI.git
cd VoxCPM-Japanese-WebUI
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1
```

詳しい手順:

- README: https://github.com/ShibaPapaMikami/VoxCPM-Japanese-WebUI
- セットアップ: https://github.com/ShibaPapaMikami/VoxCPM-Japanese-WebUI/blob/main/README_SETUP_JA.md
- スモークテスト: https://github.com/ShibaPapaMikami/VoxCPM-Japanese-WebUI/blob/main/docs/SMOKE_TEST_JA.md

### 注意

声のクローン機能は、自分の声、または本人から明示的な許可を得た音声だけに使用してください。
生成音声を実在人物の発言として偽装したり、第三者の権利を侵害する用途には使わないでください。
````

## 公開後チェック

- ReleaseページからREADME、セットアップ手順、スモークテスト手順に移動できる
- 添付zipを展開して、`pretrained_models/`、`outputs/`、`external/` が含まれていない
- 別フォルダに展開して `scripts\check_setup.ps1` が実行できる
- exeを添付した場合は、別フォルダに展開して `VoxCPM_WebUI.exe -Port 8808` が起動できる
- Issueテンプレートが表示される

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

非エンジニア向けには、GitHub Releasesのリンクと `README_SETUP_JA.md` だけを案内すると迷いにくいです。
