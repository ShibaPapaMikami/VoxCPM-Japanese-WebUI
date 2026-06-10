# JP Voice Studio セットアップ

JP Voice Studioは、OpenBMB/VoxCPM をベースに、日本語UI、声のデザイン履歴、WAVダウンロード、多言語選択、高精度クローン補助などを追加した音声生成・声クローン統合ツールです。任意の追加エンジンとして Irodori-TTS と VoiceDesignCloner連携（Qwen3-TTS・簡易）も利用できます。

## 対象環境

- Windows 10/11
- NVIDIA GPU 推奨
- CUDA 12系対応ドライバ
- Python 3.10 または 3.11
- Git
- uv

uv がない場合:

```powershell
winget install --id Astral-sh.UV
```

## 最短起動

PowerShellで以下を実行します。

```powershell
git clone https://github.com/ShibaPapaMikami/VoxCPM-Japanese-WebUI.git
cd VoxCPM-Japanese-WebUI
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1
```

起動後、ブラウザで開きます。

```text
http://127.0.0.1:8808/
```

セットアップ後の最小確認は [最小スモークテスト手順](./docs/SMOKE_TEST_JA.md) を参照してください。

## セットアップ診断

セットアップ後に起動できない場合、またはVoxCPM2 / Irodori-TTS / Qwen3-TTS / CUDA / モデルキャッシュ / 8808番ポートの状態をまとめて確認したい場合は、以下を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_setup.ps1
```

LAN公開で使う場合は、ホストとポートも指定できます。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_setup.ps1 -HostAddress 0.0.0.0 -Port 8808
```

結果は `PASS` / `WARN` / `FAIL` で表示されます。`FAIL` がある場合は表示された `fix:` の手順を実行してください。`WARN` は動作可能な場合もありますが、必要に応じて修復してください。

同じLAN内の別端末から使う場合は、起動PCのIPアドレスを使います。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1 -SkipBaseSetup -HostAddress 0.0.0.0
```

```text
http://<起動PCのIPアドレス>:8808/
```

必要ならWindowsファイアウォールを許可します。既定ではプライベートネットワークだけを許可します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1 -SkipBaseSetup -NoLaunch -AllowFirewall
```

## 2回目以降の起動

セットアップ済みなら、依存関係の再インストールを省略できます。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1 -SkipBaseSetup
```

またはランチャーだけ使います。

```powershell
.\VoxCPM_WebUI.cmd
```

## 任意エンジンもまとめて入れる場合

Irodori-TTSとQwen3-TTSも一緒にセットアップする場合は、初回に以下を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1 -AllEngines
```

個別に追加する場合は、`-WithIrodori` または `-WithQwen3` を指定します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1 -WithIrodori
powershell -ExecutionPolicy Bypass -File scripts\setup_all_windows.ps1 -WithQwen3
```

## モデルについて

初回起動時または初回生成時に VoxCPM2 のモデルを取得します。モデルは `pretrained_models/` に保存されます。

`pretrained_models/` は巨大ファイルなのでGitHubには含めません。

## Irodori-TTSを追加する場合

Irodori-TTSは日本語特化の任意エンジンです。VoxCPM2とは依存関係が異なるため、別フォルダ `external/Irodori-TTS/` にセットアップします。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_irodori_tts.ps1
```

完了後にWeb UIを再起動し、画面上部の「音声エンジン」で `Irodori-TTS（日本語特化・実験）` を選びます。

Irodori-TTSは日本語専用です。多言語生成や高精度クローンを使う場合は `VoxCPM2（総合）` を選んでください。

## VoiceDesignCloner連携（Qwen3-TTS・簡易）を追加する場合

VoiceDesignCloner連携（Qwen3-TTS・簡易）は、Voice-Design-Cloner本体を組み込むものではなく、Qwen3-TTSワークフローを参考にしたJP Voice Studio内の簡易連携です。多言語の声デザイン、生成数指定による声ガチャ、参照音声+文字起こしによる簡易クローン、選んだ声での簡易コーパス一括音声化、リサンプル、esd.list生成、Irodori-TTS LoRA学習データ準備、LoRA学習実行入口に対応します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen3_tts.ps1
```

完了後にWeb UIを再起動し、画面上部の「音声エンジン」で `VoiceDesignCloner連携（Qwen3-TTS・簡易）` を選びます。

Qwen3-TTSの声のクローンでは、参照音声の文字起こしが必要です。Qwen3-TTSで作った声のデザイン履歴は、WAVの横に `.txt` を保存するため、そのまま履歴再利用できます。コーパス一括音声化では、1行1文のテキストから `raw/*.wav` と `Neutral.txt` を生成します。必要に応じて `resampled/`、`esd.list`、Irodori-TTS LoRA学習用の `lora_data/lab/{話者}/{感情}` も作成できます。LoRA学習実行は既定でドライランです。実学習ではGPUと時間を使うため、少ないステップ数から試してください。

LoRA学習後は、音声エンジンで `Irodori-TTS（日本語特化・実験）` を選び、声のデザインまたは声のクローンタブの「Irodori LoRAアダプタ」から学習済みアダプタを選択できます。

Voice-Design-Cloner本体と同等の全自動ワークフロー、Style-Bert-VITS2向け完全自動配置、実運用向けに調整済みのLoRAファインチューン一式は、現時点ではまだ未統合または簡易対応です。

## 主な使い方

- 声のデザイン: 参照音声なしで声を作る
- 声ガチャ: VoiceDesignCloner連携で複数候補を生成して試聴する
- コーパス一括音声化: VoiceDesignCloner連携で1行1文のテキストをまとめてWAV化する
- リサンプル・esd.list生成: 生成したコーパスをStyle-Bert-VITS2向けの入口形式に整える
- LoRA学習データ準備: 生成したコーパスをIrodori-TTSの学習用lab形式に変換する
- LoRA学習実行: ドライランでコマンドを確認し、必要に応じて実学習を開始する
- LoRA推論: Irodori-TTS選択時に学習済みLoRAアダプタを適用して生成する
- LoRA管理: 学習済みアダプタ一覧の更新、保存フォルダ表示、学習データ準備後のlabフォルダ自動入力
- 声のデザイン履歴から再利用: 作った声を参照音声として別セリフに使う
- 声のクローン: 参照音声の声質で別テキストを読む
- 高精度クローン: 参照音声と文字起こしを使って再現度を上げる
- 発話言語: テキストの発話言語を指定する
- 音声エンジン: VoxCPM2、Irodori-TTS、VoiceDesignCloner連携（Qwen3-TTS・簡易）を切り替える
- 単語アクセントを指定: `イチゴ=語尾上げ` のように単語ごとの読み方を補助する
- 記号で読み方を調整: `「」`、`、`、`……`、`！`、`？` を本文に挿入する

## 注意

VoxCPMはApache-2.0ライセンスです。モデル重みとコードもApache-2.0として公開されていますが、声のクローン機能を使う場合は、本人の許可がある音声だけを利用してください。

生成音声を実在人物の発言として偽装したり、第三者の権利を侵害する用途には使わないでください。
