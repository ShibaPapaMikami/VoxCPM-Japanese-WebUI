# サンプル生成と画面イメージ

このページは、初めて使う人がJP Voice Studioの雰囲気をつかむためのサンプルです。生成音声そのものはリポジトリに含めません。各環境で同じ文を生成して確認してください。

![JP Voice Studio画面イメージ](./images/jp-voice-studio-overview.svg)

## サンプル音声の作り方

生成音声は `outputs/` に保存されます。`outputs/` は `.gitignore` の対象なので、GitHubへ誤って含まれにくくなっています。

## VoxCPM2

Web UI:

- 音声エンジン: `VoxCPM2（総合）`
- タブ: `声のデザイン`
- 発話言語: `日本語`
- 声の指示: `落ち着いた日本語の男性ナレーション。聞き取りやすく、少しゆっくり話す。`
- 読み上げテキスト: `JP Voice Studioのサンプル音声を生成しています。`

CLI:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_smoke_jp.ps1 `
  -Text "JP Voice Studioのサンプル音声を生成しています。" `
  -Control "calm Japanese male narration" `
  -Output "outputs\sample_voxcpm2.wav" `
  -InferenceTimesteps 4
```

## Irodori-TTS

Irodori-TTSをセットアップ済みの場合だけ確認します。

- 音声エンジン: `Irodori-TTS（日本語特化・実験）`
- タブ: `声のデザイン`
- 発話言語: `日本語`
- 年齢: `大人`
- 性別: `女性` または `男性`
- 読み上げテキスト: `JP Voice Studioの日本語音声サンプルを生成しています。`

## VoiceDesignCloner連携（Qwen3-TTS・簡易）

Qwen3-TTS連携をセットアップ済みの場合だけ確認します。

- 音声エンジン: `VoiceDesignCloner連携（Qwen3-TTS・簡易）`
- タブ: `声のデザイン`
- 発話言語: `日本語`
- 声の指示: `低めで落ち着いた日本語ナレーション。自然で聞き取りやすい。`
- 読み上げテキスト: `JP Voice Studioのキュウウェン音声サンプルを生成しています。`
- 生成数: `1`

## サンプルを公開する場合の注意

- 自分の声、社内で利用許諾を得た音声、または本人から明示的な許可を得た音声だけを使ってください。
- 生成音声を実在人物の発言として偽装しないでください。
- サンプル音声をGitHubへ置く場合は、権利確認済みの短い音声だけにしてください。
- 社内音声、参照音声、文字起こし、顧客データは公開リポジトリに含めないでください。
