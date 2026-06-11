# Windows実行ファイルの作り直し手順

JP Voice Studioは、通常は `VoxCPM_WebUI.cmd` から起動できます。`VoxCPM_WebUI.exe` は、その `.cmd` と同じように `scripts\launch_webui.ps1` を呼び出すだけの軽いWindowsランチャーです。

## 重要な前提

- `VoxCPM_WebUI.exe` はGit管理対象外です。
- Releaseに含める場合は、必ずこの手順で作り直してから添付してください。
- `pretrained_models/`、`outputs/`、`external/`、`.venv/` はexeに含まれません。
- exeはモデルや依存関係を内包しません。初回セットアップは別途必要です。

## まずはcmdを推奨

配布時は、基本的には以下のどちらかを案内するのが安全です。

```powershell
.\JPVoiceStudio_Setup.cmd
```

または:

```powershell
.\VoxCPM_WebUI.cmd
```

exeは、ダブルクリック起動を分かりやすくしたい場合だけ作成します。

## ビルド方法

PowerShellをリポジトリのフォルダで開きます。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_launcher.ps1
```

成功すると、リポジトリ直下に `VoxCPM_WebUI.exe` が作成されます。

既存ファイルを上書きしたくない場合:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_launcher.ps1 -NoOverwrite
```

Release確認用に `dist/` へ出す場合:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_launcher.ps1 -Output dist\VoxCPM_WebUI.exe
```

## ビルドに必要なもの

このスクリプトは `csc.exe` を使います。通常のWindowsでは、以下のような場所にあります。

```text
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe
```

見つからない場合は、.NET Framework Developer Pack または Visual Studio Build Tools をインストールしてください。

## 動作確認

ビルド後、以下を実行します。

```powershell
.\VoxCPM_WebUI.exe -Port 8808
```

確認すること:

- PowerShellが起動し、`scripts\launch_webui.ps1` が呼ばれる
- ブラウザで `http://127.0.0.1:8808/` が開ける
- `JP Voice Studio` が表示される

終了後、必要なら最小スモークテストも実行します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_setup.ps1
```

## Releaseに添付する場合

添付前に以下を確認します。

```powershell
Get-Item .\VoxCPM_WebUI.exe | Select-Object FullName,Length,LastWriteTime
```

Release本文には、exeだけではセットアップが完了しないことを明記してください。

```text
VoxCPM_WebUI.exeは起動用ランチャーです。初回利用時はREADME_SETUP_JA.mdに従ってセットアップしてください。
```

## トラブルシュート

- `csc.exe was not found`: .NET Framework Developer Pack または Visual Studio Build Tools を入れてください。
- `Missing launcher script`: exeと同じフォルダに `scripts\launch_webui.ps1` がありません。リポジトリ全体を展開してください。
- 起動してすぐ閉じる: `VoxCPM_WebUI.cmd` で起動し、PowerShell上のエラーを確認してください。
- セキュリティ警告が出る: 自作exeのためWindows Defender SmartScreenが警告する場合があります。社内配布では署名や配布経路のルールに従ってください。
