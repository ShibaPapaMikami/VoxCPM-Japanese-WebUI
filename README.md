# JPVoxCPM WebUI

<p align="center">
  <b>日本語</b> | <a href="./README_en.md">English</a> | <a href="./README_zh.md">中文</a>
</p>

日本語で使いやすい VoxCPM2 音声生成・声クローンWeb UIです。  
OpenBMB/VoxCPMをベースに、日本語UI、Windows CUDAセットアップ、声のデザイン履歴、WAVダウンロード、多言語発話、声のクローン操作を追加しています。

元プロジェクト: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)

## 主な機能

- 声のデザイン: テキスト指示だけで声を作成
- 声のデザイン履歴: 作成済みの声を参照して別セリフを生成
- 声のクローン: 参照音声の声質で読み上げ
- 高精度クローン: 参照音声と文字起こしを使って再現性を向上
- 多言語発話: 発話言語を選んで読み上げ
- 記号による読み方調整: 強調、間、疑問、語尾などを指定
- 単語ごとのアクセント指定: 平坦、語尾上げ、頭高、中高、尾高をUIから追加
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

## 最短起動

PowerShellでリポジトリ直下に移動して実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1
```

起動後、ブラウザで開きます。

```text
http://127.0.0.1:8808/
```

2回目以降、セットアップ済みなら依存関係の再インストールを省略できます。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1 -SkipSetup
```

または、ランチャーを使います。

```powershell
.\VoxCPM_WebUI.cmd
```

詳しい手順は [README_SETUP_JA.md](README_SETUP_JA.md) を参照してください。

## 社内LANから使う場合

同じLAN内の別端末からアクセスする場合は、起動PCのIPアドレスを使います。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1 -HostAddress 0.0.0.0
```

```text
http://<起動PCのIPアドレス>:8808/
```

必要に応じてWindowsファイアウォールで8808番ポートを許可します。既定ではプライベートネットワークだけを許可します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\allow_firewall_8808.ps1
```

## モデルについて

初回起動時、または初回生成時にVoxCPM2のモデルを取得します。  
モデルは `pretrained_models/` に保存されます。

`pretrained_models/` は大きなファイルを含むため、このリポジトリには含めていません。

## 公開・配布について

このリポジトリはApache-2.0ライセンスのOpenBMB/VoxCPMをベースにしています。  
公開時は `LICENSE` を残し、元プロジェクトへのリンクを維持してください。

GitHub公開や社内配布の確認項目は [docs/GITHUB_RELEASE_JA.md](docs/GITHUB_RELEASE_JA.md) にまとめています。

## 注意

声のクローン機能を使う場合は、本人の許可がある音声だけを利用してください。  
生成音声を実在人物の発言として偽装したり、第三者の権利を侵害する用途には使わないでください。

## 参考リンク

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2 on Hugging Face](https://huggingface.co/openbmb/VoxCPM2)
- [VoxCPM2 on ModelScope](https://modelscope.cn/models/OpenBMB/VoxCPM2)
