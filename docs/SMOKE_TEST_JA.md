# JP Voice Studio 最小スモークテスト

セットアップ後に「この環境で最低限動くか」を確認するための短い手順です。まず共通診断を実行し、その後に使いたい音声エンジンだけ確認してください。

## 1. 共通診断

PowerShellをこのリポジトリのフォルダで開きます。

```powershell
cd path\to\VoxCPM-Japanese-WebUI
powershell -ExecutionPolicy Bypass -File scripts\check_setup.ps1
```

社内LANなど、別PCからアクセスする設定も確認したい場合は次を使います。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_setup.ps1 -HostAddress 0.0.0.0 -Port 8808
```

目安:

- `FAIL=0` なら基本セットアップは通っています。
- `WARN` は任意エンジン未導入やファイアウォール確認など、用途によって許容できる項目です。
- `FAIL` が出た場合は、表示された `fix` の内容を優先して直してください。

## 2. Web UI起動確認

```powershell
powershell -ExecutionPolicy Bypass -File scripts\launch_webui.ps1 -Port 8808
```

ブラウザで次を開きます。

```text
http://127.0.0.1:8808/
```

確認すること:

- 画面上部に `JP Voice Studio` が表示される
- 音声エンジンで `VoxCPM2（総合）`、`Irodori-TTS（日本語特化・実験）`、`VoiceDesignCloner連携（Qwen3-TTS・簡易）` を切り替えられる
- Irodori-TTS / Qwen3-TTSを未セットアップの場合、案内文が表示され、UI全体は壊れない

## 3. VoxCPM2の生成確認

まずは短い文で確認します。初回はモデル読み込みに時間がかかります。

Web UIで以下を入力します。

- 音声エンジン: `VoxCPM2（総合）`
- タブ: `声のデザイン`
- 発話言語: `日本語`
- 声の指示: `落ち着いた日本語の男性ナレーション。聞き取りやすく、少しゆっくり話す。`
- 読み上げテキスト: `今日は音声生成の動作確認をしています。`

確認すること:

- エラーなく音声が生成される
- プレイヤーで再生できる
- `WAVダウンロード` にファイルが表示される
- 保存先フォルダにもWAVが作成される

CLIだけで確認する場合は、次のスクリプトも使えます。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_smoke_jp.ps1 `
  -Text "今日は音声生成の動作確認をしています。" `
  -Control "calm Japanese male narration" `
  -Output "outputs\smoke_voxcpm2.wav" `
  -InferenceTimesteps 4
```

生成された `outputs\smoke_voxcpm2.wav` を再生できれば、VoxCPM2の最小生成は通っています。

## 4. Irodori-TTSの生成確認

Irodori-TTSを使う場合だけ確認します。未セットアップの場合は、先に以下を実行してください。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_irodori_tts.ps1
```

Web UIで以下を入力します。

- 音声エンジン: `Irodori-TTS（日本語特化・実験）`
- タブ: `声のデザイン`
- 発話言語: `日本語`
- 年齢: `大人`
- 性別: `女性` または `男性`
- 読み上げテキスト: `今日は日本語音声の動作確認をしています。`

確認すること:

- ヘッダーがIrodori-TTSロゴに切り替わる
- Irodori-TTSで使えない多言語・高精度クローン系UIが前面に出ない
- エラーなく音声が生成される
- LoRAアダプタ未選択でも通常生成できる

## 5. VoiceDesignCloner連携（Qwen3-TTS・簡易）の生成確認

Qwen3-TTS連携を使う場合だけ確認します。未セットアップの場合は、先に以下を実行してください。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen3_tts.ps1
```

Web UIで以下を入力します。

- 音声エンジン: `VoiceDesignCloner連携（Qwen3-TTS・簡易）`
- タブ: `声のデザイン`
- 発話言語: `日本語`
- 声の指示: `落ち着いた日本語の男性ナレーション。聞き取りやすく、少しゆっくり話す。`
- 読み上げテキスト: `今日はキュウウェン音声の動作確認をしています。`
- 生成数: `1`

確認すること:

- ヘッダーがVoiceDesignClonerロゴに切り替わる
- Qwen3-TTSの簡易連携である説明が表示される
- エラーなく音声が生成される
- 生成数を `2` 以上にした場合、候補一覧が表示される

## 6. 声のクローン確認

本人の許可がある参照音声だけを使ってください。社内テストでは、10秒前後の明瞭な録音を推奨します。

VoxCPM2:

- タブ: `声のクローン`
- 参照音声: 許諾済みWAV
- 読み上げテキスト: `今日は声のクローン機能の確認をしています。`

Qwen3-TTS:

- タブ: `声のクローン`
- 参照音声: 許諾済みWAV
- 参照音声の文字起こし: 参照音声で実際に話している文
- 読み上げテキスト: `今日は声のクローン機能の確認をしています。`

Irodori-TTS:

- タブ: `声のクローン`
- 参照音声: 許諾済みWAV
- 読み上げテキスト: `今日は日本語音声の確認をしています。`

確認すること:

- 参照音声が短すぎる、または文字起こしが不足している場合に分かりやすいエラーが出る
- 高精度クローン非対応エンジンでは、高精度クローンタブではなく声のクローンタブを案内する

## 7. よくある失敗

- Web UIが開かない: `scripts\check_setup.ps1 -Port 8808` でポート使用状況を確認します。
- 別PCから開けない: `scripts\allow_firewall_8808.ps1` を管理者PowerShellで実行し、起動時に `-HostAddress 0.0.0.0` を使います。
- Irodori-TTSが選べるが生成できない: `external\Irodori-TTS` とIrodori側の `.venv` が存在するか確認します。
- Qwen3-TTSの声クローンで失敗する: 参照音声の文字起こしが必須です。
- 生成音声やモデルをGitに入れそうになる: `outputs/`、`pretrained_models/`、`external/` は `.gitignore` の対象です。

## 8. 報告時に添える情報

GitHub Issueや社内問い合わせでは、以下を添えると原因を追いやすくなります。

- 使った音声エンジン
- 操作したタブ
- 入力した読み上げテキスト
- エラー全文
- `scripts\check_setup.ps1` の結果
- 可能ならスクリーンショット
