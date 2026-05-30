# VoxCPM 日本語Web UI セットアップ

このリポジトリは OpenBMB/VoxCPM をベースに、日本語UI、声のデザイン履歴、WAVダウンロード、多言語選択、高精度クローン補助などを追加した社内向けWeb UIです。

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

PowerShellでリポジトリ直下に移動して実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1
```

起動後、ブラウザで開きます。

```text
http://127.0.0.1:8808/
```

同じLAN内の別端末から使う場合は、起動PCのIPアドレスを使います。

```text
http://<起動PCのIPアドレス>:8808/
```

必要ならWindowsファイアウォールを許可します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\allow_firewall_8808.ps1
```

## 2回目以降の起動

セットアップ済みなら、依存関係の再インストールを省略できます。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_and_launch_windows_cuda.ps1 -SkipSetup
```

またはランチャーだけ使います。

```powershell
.\VoxCPM_WebUI.cmd
```

## モデルについて

初回起動時または初回生成時に VoxCPM2 のモデルを取得します。モデルは `pretrained_models/` に保存されます。

`pretrained_models/` は巨大ファイルなのでGitHubには含めません。

## 主な使い方

- 声のデザイン: 参照音声なしで声を作る
- 声のデザイン履歴から再利用: 作った声を参照音声として別セリフに使う
- 声のクローン: 参照音声の声質で別テキストを読む
- 高精度クローン: 参照音声と文字起こしを使って再現度を上げる
- 発話言語: テキストの発話言語を指定する
- 単語アクセントを指定: `イチゴ=語尾上げ` のように単語ごとの読み方を補助する
- 記号で読み方を調整: `「」`、`、`、`……`、`！`、`？` を本文に挿入する

## 注意

VoxCPMはApache-2.0ライセンスです。モデル重みとコードもApache-2.0として公開されていますが、声のクローン機能を使う場合は、本人の許可がある音声だけを利用してください。

生成音声を実在人物の発言として偽装したり、第三者の権利を侵害する用途には使わないでください。
