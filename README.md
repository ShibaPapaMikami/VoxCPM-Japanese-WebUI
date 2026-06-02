# JP Voice Studio

<p align="center">
  <b>日本語</b> | <a href="./README_en.md">English</a> | <a href="./README_zh.md">中文</a>
</p>

日本語で使いやすい音声生成・声クローン統合ツールです。<br>
OpenBMB/VoxCPMをベースに、日本語UI、Windows CUDAセットアップ、声のデザイン履歴、WAVダウンロード、多言語発話、声のクローン操作を追加しています。任意の追加エンジンとして Irodori-TTS と Qwen3-TTS も利用できます。

元プロジェクト: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)

## 主な機能

- 声のデザイン: テキスト指示だけで声を作成
- 声ガチャ: VoiceDesignCloner連携で複数候補を連続生成して試聴
- コーパス一括音声化（簡易）: VoiceDesignCloner連携で選んだ声を使い、1行1文のテキストをまとめてWAV化
- Style-Bert-VITS2向け前処理（簡易）: コーパスの `raw/*.wav` をリサンプルし、`Neutral.txt` から `esd.list` を生成
- 声のデザイン履歴: 作成済みの声を参照して別セリフを生成
- 声のクローン: 参照音声の声質で読み上げ
- 高精度クローン: 参照音声と文字起こしを使って再現性を向上
- 参照音声録音ガイド: 推奨秒数、録音原稿、文字起こし欄への反映を補助
- 多言語発話: 発話言語を選んで読み上げ
- 記号による読み方調整: 強調、間、疑問、語尾などを指定
- 単語ごとのアクセント指定: 平坦、語尾上げ、頭高、中高、尾高をUIから追加
- 音声エンジン切替: VoxCPM2に加えて、Irodori-TTSとVoiceDesignCloner連携（Qwen3-TTS・簡易）を選択可能
- VoiceDesignCloner連携（簡易）: Voice-Design-Clonerを参考に、Qwen3-TTSの声デザイン・声ガチャ・参照音声クローン・簡易コーパス生成・リサンプル・esd.list生成をWeb UIから実行
- WAVダウンロード: 生成した音声をそのまま保存
- Windows向け簡単起動スクリプト

## 対象環境

- Windows 10 / 11
- NVIDIA GPU推奨
- CUDA 12系対応ドライバ
- Python 3.10 または 3.11
- Git
- uv

uvがない場合:

```powershell
winget install --id Astral-sh.UV
```

## クイックスタート

PowerShellで以下を実行します。初回は依存関係とVoxCPM2モデルの取得に時間がかかります。

```powershell
git clone https://github.com/ShibaPapaMikami/VoxCPM-Japanese-WebUI.git
cd VoxCPM-Japanese-WebUI
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1
```

起動後、ブラウザで開きます。

```text
http://127.0.0.1:8808/
```

### 2回目以降

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1 -SkipSetup
```

または、ランチャーを使います。

```powershell
.\VoxCPM_WebUI.cmd
```

### Irodori-TTSも使う場合

VoxCPM2だけ使う場合は不要です。日本語特化エンジンのIrodori-TTSも使う場合だけ、追加で実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_irodori_tts.ps1
```

完了後にWeb UIを再起動し、画面上部の「音声エンジン」で `Irodori-TTS（日本語特化・実験）` を選びます。

### Qwen3-TTSも使う場合

VoxCPM2だけ使う場合は不要です。Voice-Design-Clonerで採用されているQwen3-TTS系の声デザイン・声クローンも使う場合だけ、追加で実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_qwen3_tts.ps1
```

完了後にWeb UIを再起動し、画面上部の「音声エンジン」で `VoiceDesignCloner連携（Qwen3-TTS・簡易）` を選びます。

Qwen3-TTSの声のクローンでは、参照音声と、その参照音声で話している文字起こしが必要です。Qwen3-TTSで声のデザインを生成した履歴は、WAVの横に参照テキストを保存するため、そのまま「履歴の声で生成」に使えます。

セットアップ時に `SoX could not be found` という警告が出ることがあります。`qwen_tts import ok` が表示されていれば導入自体は完了していますが、生成時に音声前処理エラーが出る場合はSoXの追加インストールを検討してください。

### LAN内の別端末から使う場合

同じLAN内の別端末からアクセスする場合は、起動PCで以下のように起動します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1 -HostAddress 0.0.0.0
```

別端末のブラウザで開きます。

```text
http://<起動PCのIPアドレス>:8808/
```

必要に応じてWindowsファイアウォールで8808番ポートを許可します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\allow_firewall_8808.ps1
```

詳しい手順は [README_SETUP_JA.md](README_SETUP_JA.md) を参照してください。

## モデルについて

初回起動時、または初回生成時にVoxCPM2のモデルを取得します。  
モデルは `pretrained_models/` に保存されます。

`pretrained_models/` は大きなファイルを含むため、このリポジトリには含めていません。

### Irodori-TTSを使う場合

Irodori-TTSは任意の追加エンジンです。依存関係の競合を避けるため、このWeb UI本体とは別フォルダ `external/Irodori-TTS/` にセットアップします。手順は上の「Irodori-TTSも使う場合」を参照してください。

Irodori-TTSは日本語専用です。多言語、VoxCPM2の高精度クローン、自由文による細かな声の指示を使う場合は、既定の `VoxCPM2（総合）` を使ってください。

### VoiceDesignCloner連携（Qwen3-TTS・簡易）を使う場合

VoiceDesignCloner連携（Qwen3-TTS・簡易）は任意の追加エンジンです。`scripts/setup_qwen3_tts.ps1` で、このWeb UIの `.venv` に `qwen-tts` と `sentencepiece` を追加します。

Qwen3-TTSは10言語（日本語、英語、中国語、韓国語、ドイツ語、フランス語、ロシア語、ポルトガル語、スペイン語、イタリア語）に対応しています。このWeb UIでは、声のデザイン、声ガチャ、簡易クローン、選んだ声での簡易コーパス一括音声化、リサンプル、esd.list生成で利用できます。

Voice-Design-Cloner本体の以下の機能は、現時点ではまだ統合していません。

- Irodori-TTS LoRAファインチューン
- Style-Bert-VITS2向け完全自動配置

## 公開・配布について

このリポジトリはApache-2.0ライセンスのOpenBMB/VoxCPMをベースにし、任意エンジンとしてMITライセンスのIrodori-TTS、Apache-2.0ライセンスのQwen3-TTSを利用できます。<br>
公開時は `LICENSE` を残し、元プロジェクトへのリンクと第三者ライセンス表記を維持してください。

GitHub公開や社内配布の確認項目は [docs/GITHUB_RELEASE_JA.md](docs/GITHUB_RELEASE_JA.md) にまとめています。<br>
主な第三者プロジェクト、モデル、商用利用、声クローンの注意事項は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を確認してください。

## 注意

声のクローン機能を使う場合は、本人の許可がある音声だけを利用してください。  
生成音声を実在人物の発言として偽装したり、第三者の権利を侵害する用途には使わないでください。

## 参考リンク

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2 on Hugging Face](https://huggingface.co/openbmb/VoxCPM2)
- [VoxCPM2 on ModelScope](https://modelscope.cn/models/OpenBMB/VoxCPM2)
- [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- [Irodori-TTS-500M-v3 on Hugging Face](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
- [reinehonoka/Voice-Design-Cloner](https://github.com/reinehonoka/Voice-Design-Cloner)
- [Qwen3-TTS-12Hz-1.7B-VoiceDesign on Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
