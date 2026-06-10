import os
import re
import sys
import json
import logging
import shutil
import subprocess
import time
import numpy as np
import gradio as gr
from gradio import processing_utils
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path
from datetime import datetime
from uuid import uuid4

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.utils import resolve_runtime_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    '`Target Text`: *"I never asked you to stay… It\'s not like I care or anything. '
    'But… why does it still hurt so much now that you\'re gone?"*  \n\n'
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,       # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

# ---------- Japanese UI override ----------

_USAGE_INSTRUCTIONS_JA = (
    "**VoxCPM2 - 音声生成モード**\n\n"
    "**声のデザイン**  \n"
    "参照音声なしで新しい声を作れます。声の特徴、年齢、雰囲気、感情、話す速さなどを"
    "「声の指示」に書くと、その内容に合わせて音声を生成します。\n\n"
    "**声のクローン**  \n"
    "参照音声をアップロードすると、その声質をもとに読み上げます。声の指示を追加すると、"
    "明るさ、落ち着き、速さなどを調整できます。\n\n"
    "**高精度クローン**  \n"
    "参照音声とその文字起こしを使って、声色や話し方をより細かく再現します。"
    "このモードでは声の指示は使われません。"
)

_EXAMPLES_FOOTER_JA = (
    "---\n"
    "**声の指示例**\n\n"
    "**落ち着いたナレーション**  \n"
    "`声の指示`: `落ち着いた日本語の男性ナレーション。聞き取りやすく、少しゆっくり話す。`  \n"
    "`読み上げテキスト`: `こんにちは。VoxCPM2の日本語音声生成テストです。`\n\n"
    "**やさしい案内音声**  \n"
    "`声の指示`: `やさしい女性の声。明るく親しみやすい案内口調。`  \n"
    "`読み上げテキスト`: `本日はご利用ありがとうございます。次の画面で内容をご確認ください。`\n\n"
    "**元気なキャラクター声**  \n"
    "`声の指示`: `若く元気な声。テンポは少し速めで、楽しそうに話す。`  \n"
    "`読み上げテキスト`: `準備はできた？それでは新しい音声を作ってみよう！`"
)

_I18N_JA = {
    "reference_audio_label": "参照音声（任意。クローンしたい声をアップロード）",
    "show_prompt_text_label": "高精度クローンモード（文字起こしを使う）",
    "show_prompt_text_info": "参照音声を文字起こしして、声色や話し方をより細かく再現します。有効にすると声の指示は無効になります。",
    "prompt_text_label": "参照音声の文字起こし（自動入力・編集可）",
    "prompt_text_placeholder": "参照音声の文字起こしがここに入ります...",
    "control_label": "声の指示（任意）",
    "control_placeholder": "例: 落ち着いた日本語の男性ナレーション / やさしい女性の声 / 明るく速めの話し方",
    "target_text_label": "読み上げテキスト",
    "generate_btn": "音声を生成",
    "generated_audio_label": "生成された音声",
    "advanced_settings_title": "詳細設定",
    "ref_denoise_label": "参照音声のノイズ除去",
    "ref_denoise_info": "クローン前にZipEnhancerで参照音声を補正します。",
    "normalize_label": "テキスト正規化",
    "normalize_info": "数字、日付、省略表記などを読み上げ向けに整えます。",
    "cfg_label": "CFG（指示への追従度）",
    "cfg_info": "大きいほど指示や参照音声に寄せ、小さいほど生成の自由度が上がります。",
    "dit_steps_label": "LocDiT生成ステップ数",
    "dit_steps_info": "ステップ数を増やすと品質が上がる場合がありますが、生成は遅くなります。",
    "usage_instructions": _USAGE_INSTRUCTIONS_JA,
    "examples_footer": _EXAMPLES_FOOTER_JA,
}

_I18N_TRANSLATIONS = {
    "en": _I18N_JA,
    "ja": _I18N_JA,
    "ja-JP": _I18N_JA,
    "zh": _I18N_JA,
    "zh-CN": _I18N_JA,
    "zh-Hans": _I18N_JA,
}
I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = "こんにちは。VoxCPM2の日本語音声生成テストです。自然で聞き取りやすい音声を生成します。"

_CUSTOM_CSS = """
:root {
    --jp-surface: #ffffff;
    --jp-page: #f6f7f9;
    --jp-border: #d9e0ea;
    --jp-border-strong: #c5cfdd;
    --jp-text: #172033;
    --jp-muted: #637083;
    --jp-accent: #2563eb;
    --jp-accent-soft: #e8f0ff;
    --jp-warm: #f27a1a;
}
body,
.gradio-container {
    background: var(--jp-page) !important;
    color: var(--jp-text);
}
.gradio-container {
    max-width: 1360px !important;
    padding: 20px 24px 36px !important;
}
.logo-container {
    margin: 0.25rem 0 1rem 0;
    padding: 1rem 1.15rem;
    border: 1px solid var(--jp-border);
    border-radius: 8px;
    background: var(--jp-surface);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    box-shadow: 0 1px 2px rgba(18, 25, 38, 0.04);
}
.logo-container img {
    height: 52px;
    width: auto;
    max-width: 200px;
    display: block;
}
.app-brand {
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.text-logo {
    border: 1px solid var(--jp-border);
    border-left: 6px solid var(--jp-warm);
    border-radius: 8px;
    padding: 0.55rem 0.8rem;
    background: #fff7ef;
    color: var(--jp-text);
    font-weight: 800;
    letter-spacing: 0;
    line-height: 1.1;
}
.text-logo span {
    display: block;
    color: #f27a1a;
    font-size: 0.78rem;
    font-weight: 700;
    margin-top: 0.2rem;
}
.brand-copy h1 {
    margin: 0;
    font-size: 1.55rem;
    line-height: 1.2;
}
.brand-copy p {
    margin: 0.25rem 0 0 0;
    color: var(--jp-muted);
}
.engine-pill {
    border-radius: 999px;
    border: 1px solid var(--jp-border);
    padding: 0.45rem 0.75rem;
    color: var(--jp-muted);
    background: #f8fafc;
    white-space: nowrap;
    font-weight: 700;
}
.gradio-container .tabs {
    gap: 0.35rem;
}
.gradio-container .tab-nav,
.gradio-container div[role="tablist"] {
    border-bottom: 1px solid var(--jp-border) !important;
    gap: 0.25rem !important;
}
.gradio-container button[role="tab"] {
    border-radius: 6px 6px 0 0 !important;
    padding: 0.65rem 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
}
.gradio-container button[role="tab"][aria-selected="true"] {
    background: var(--jp-surface) !important;
    color: var(--jp-accent) !important;
    border-color: var(--jp-border) !important;
    box-shadow: inset 0 -2px 0 var(--jp-accent);
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .accordion {
    border-radius: 8px !important;
}
.gradio-container .block {
    border-color: var(--jp-border) !important;
}
.gradio-container label,
.gradio-container .label-wrap {
    color: var(--jp-text) !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: 6px !important;
    border-color: var(--jp-border) !important;
    background: #ffffff !important;
}
.gradio-container textarea {
    line-height: 1.6 !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: var(--jp-accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}
.gradio-container button {
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
}
.gradio-container button.primary {
    background: var(--jp-accent) !important;
    border-color: var(--jp-accent) !important;
    color: #ffffff !important;
}
.gradio-container button.secondary {
    background: #f8fafc !important;
    border-color: var(--jp-border-strong) !important;
    color: var(--jp-text) !important;
}
.gradio-container button.stop {
    border-color: #f2b8b5 !important;
}
.gradio-container .info,
.gradio-container .prose p,
.gradio-container small {
    color: var(--jp-muted) !important;
}
.gradio-container .prose strong {
    color: var(--jp-text);
}
.gradio-container audio {
    border-radius: 8px;
}
.gradio-container [data-testid="block-info"] {
    color: var(--jp-muted) !important;
}
@media (max-width: 720px) {
    .gradio-container {
        padding: 12px 12px 24px !important;
    }
    .logo-container {
        align-items: flex-start;
    }
    .app-brand {
        align-items: flex-start;
    }
    .brand-copy h1 {
        font-size: 1.28rem;
    }
    .engine-pill {
        width: 100%;
    }
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_JAPANESE_UI_FIX_JS = r"""
(() => {
  const exactText = new Map([
    ["Upload file", "ファイルをアップロード"],
    ["Record audio", "録音"],
    ["Empty value", "音声はまだありません"],
    ["Adjust volume", "音量を調整"],
    ["High volume", "大音量"],
    ["No audio", "音声はまだありません"],
  ]);

  function translateValue(value) {
    if (!value) return value;
    if (exactText.has(value)) return exactText.get(value);
    if (value.startsWith("Adjust playback speed")) return "再生速度を調整";
    if (value.startsWith("Skip backwards by")) return value.replace("Skip backwards by", "").replace("seconds", "秒戻る").trim();
    if (value.startsWith("Skip forward by")) return value.replace("Skip forward by", "").replace("seconds", "秒進む").trim();
    return value;
  }

  function translateNode(node) {
    if (!(node instanceof HTMLElement)) return;

    for (const attr of ["aria-label", "title", "placeholder"]) {
      const current = node.getAttribute(attr);
      const translated = translateValue(current);
      if (translated && translated !== current) node.setAttribute(attr, translated);
    }

    if (
      node.childNodes.length === 1 &&
      node.childNodes[0].nodeType === Node.TEXT_NODE
    ) {
      const current = node.textContent.trim();
      const translated = translateValue(current);
      if (translated && translated !== current) node.textContent = translated;
    }
  }

  function translateAll() {
    document.querySelectorAll("*").forEach(translateNode);
  }

  translateAll();
  const observer = new MutationObserver(translateAll);
  observer.observe(document.body, { childList: true, subtree: true, attributes: true });
  setTimeout(() => observer.disconnect(), 30000);
})();
"""

_JAPANESE_UI_FIX_HEAD = f"<script>{_JAPANESE_UI_FIX_JS}</script>"

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

_LANGUAGE_OPTIONS = [
    ("自動（テキストから判定）", ""),
    ("日本語", "Japanese"),
    ("英語", "English"),
    ("中国語", "Chinese"),
    ("韓国語", "Korean"),
    ("フランス語", "French"),
    ("ドイツ語", "German"),
    ("スペイン語", "Spanish"),
    ("ポルトガル語", "Portuguese"),
    ("イタリア語", "Italian"),
    ("ロシア語", "Russian"),
    ("アラビア語", "Arabic"),
    ("ヒンディー語", "Hindi"),
    ("インドネシア語", "Indonesian"),
    ("タイ語", "Thai"),
    ("ベトナム語", "Vietnamese"),
    ("トルコ語", "Turkish"),
    ("ポーランド語", "Polish"),
    ("オランダ語", "Dutch"),
    ("スウェーデン語", "Swedish"),
    ("ノルウェー語", "Norwegian"),
    ("デンマーク語", "Danish"),
    ("フィンランド語", "Finnish"),
    ("ギリシャ語", "Greek"),
    ("ヘブライ語", "Hebrew"),
    ("マレー語", "Malay"),
    ("ビルマ語", "Burmese"),
    ("クメール語", "Khmer"),
    ("ラオ語", "Lao"),
    ("スワヒリ語", "Swahili"),
    ("タガログ語", "Tagalog"),
]

_LANGUAGE_HINTS = dict(_LANGUAGE_OPTIONS)
_LANGUAGE_LABELS = [label for label, _ in _LANGUAGE_OPTIONS]

_VOICE_AGE_LABELS = ["指定なし", "赤ちゃん", "子供", "若者", "大人", "老人"]
_VOICE_GENDER_LABELS = ["指定なし", "男性", "女性", "中性的"]
_VOICE_FEATURE_LABELS = [
    "明るい",
    "暗い",
    "元気",
    "落ち着いた",
    "やさしい",
    "かわいい",
    "渋い",
    "子供っぽい",
    "大人っぽい",
    "ナレーション",
    "聞き取りやすい",
    "ゆっくり",
    "早口",
]

_ENGINE_VOXCPM = "VoxCPM2（総合）"
_ENGINE_IRODORI = "Irodori-TTS（日本語特化・実験）"
_ENGINE_QWEN3 = "VoiceDesignCloner連携（Qwen3-TTS・簡易）"
_ENGINE_LABELS = [_ENGINE_VOXCPM, _ENGINE_IRODORI, _ENGINE_QWEN3]


# ---------- Model ----------

def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _build_structured_voice_prompt(
    age_label: str,
    gender_label: str,
    feature_labels: Optional[list[str]],
) -> str:
    age_tags = {
        "赤ちゃん": "baby voice, infant-like tiny voice",
        "子供": "child voice, young child timbre",
        "若者": "young adult voice",
        "大人": "adult voice, mature timbre",
        "老人": "elderly voice, aged timbre",
    }
    gender_tags = {
        "男性": "male voice, masculine timbre",
        "女性": "female voice, feminine timbre",
        "中性的": "androgynous voice, gender-neutral timbre",
    }
    feature_tags = {
        "明るい": "bright tone",
        "暗い": "dark subdued tone",
        "元気": "energetic delivery",
        "落ち着いた": "calm and composed tone",
        "やさしい": "gentle and warm tone",
        "かわいい": "cute voice",
        "渋い": "deep rich mature voice",
        "子供っぽい": "childlike speaking style",
        "大人っぽい": "adult-like composed speaking style",
        "ナレーション": "professional narrator voice",
        "聞き取りやすい": "clear articulation",
        "ゆっくり": "slightly slow speaking pace",
        "早口": "fast speaking pace",
    }
    tags: list[str] = []
    if age_label in age_tags:
        tags.append(age_tags[age_label])
    if gender_label in gender_tags:
        tags.append(gender_tags[gender_label])
    for label in feature_labels or []:
        if label in feature_tags:
            tags.append(feature_tags[label])
    if not tags:
        return ""
    return f"Voice profile: {', '.join(dict.fromkeys(tags))}."


def _combine_voice_profile_prompt(
    age_label: str,
    gender_label: str,
    feature_labels: Optional[list[str]],
    control_instruction: str,
) -> str:
    structured = _build_structured_voice_prompt(age_label, gender_label, feature_labels)
    control = (control_instruction or "").strip()
    return " ".join(part for part in (structured, control) if part)


def _engine_is_irodori(engine_label: str) -> bool:
    return (engine_label or "").startswith("Irodori-TTS")


def _engine_is_qwen3(engine_label: str) -> bool:
    label = engine_label or ""
    return label.startswith("VoiceDesignCloner連携") or label.startswith("Qwen3-TTS")


def _ensure_irodori_japanese(target_language: str) -> None:
    if target_language and target_language not in ("自動（テキストから判定）", "日本語"):
        raise ValueError("Irodori-TTSは日本語専用です。発話言語を日本語または自動にしてください。")


def _qwen3_language_name(target_language: str) -> str:
    language_name = _LANGUAGE_HINTS.get(target_language or "", "")
    language_map = {
        "Japanese": "japanese",
        "English": "english",
        "Chinese": "chinese",
        "Korean": "korean",
        "German": "german",
        "French": "french",
        "Spanish": "spanish",
        "Italian": "italian",
        "Portuguese": "portuguese",
        "Russian": "russian",
    }
    if not language_name:
        return "auto"
    qwen_language = language_map.get(language_name)
    if not qwen_language:
        raise ValueError(
            "Qwen3-TTSで選べる発話言語は、日本語・英語・中国語・韓国語・ドイツ語・"
            "フランス語・スペイン語・イタリア語・ポルトガル語・ロシア語です。"
        )
    return qwen_language


def _build_irodori_style_text(
    text: str,
    feature_labels: Optional[list[str]] = None,
    control_instruction: str = "",
) -> str:
    """Add conservative emoji style hints for Irodori while keeping the spoken text readable."""
    feature_labels = feature_labels or []
    control = control_instruction or ""
    emojis: list[str] = []
    keyword_emoji_pairs = [
        (("明るい", "元気", "楽しい", "嬉しい"), "😊"),
        (("暗い", "悲しい", "寂しい"), "😔"),
        (("かわいい", "可愛い", "やさしい", "優しい"), "🥰"),
        (("落ち着いた", "穏やか", "ナレーション"), "🙂"),
        (("驚き", "びっくり"), "😲"),
        (("怒り", "怒った"), "😠"),
    ]
    joined = " ".join(feature_labels) + " " + control
    for keywords, emoji in keyword_emoji_pairs:
        if any(keyword in joined for keyword in keywords):
            emojis.append(emoji)
    prefix = "".join(dict.fromkeys(emojis[:2]))
    return f"{prefix}{text}" if prefix else text


def _build_irodori_caption(
    age_label: str = "",
    gender_label: str = "",
    feature_labels: Optional[list[str]] = None,
    control_instruction: str = "",
) -> str:
    """Build an Irodori caption/style prompt without changing the spoken text."""
    caption_parts: list[str] = []
    if age_label and age_label != "指定なし":
        caption_parts.append(f"{age_label}の声")
    if gender_label and gender_label != "指定なし":
        caption_parts.append(f"{gender_label}の声質")
    for label in feature_labels or []:
        if label and label != "指定なし":
            caption_parts.append(label)
    control = (control_instruction or "").strip()
    if control:
        caption_parts.append(control)
    if not caption_parts:
        return ""
    return "、".join(dict.fromkeys(caption_parts)) + "。"


def _build_control_prompt(control_instruction: str) -> str:
    """Convert Japanese UI guidance into stronger VoxCPM voice-design hints."""
    control = (control_instruction or "").strip()
    control = re.sub(r"[()（）]", "", control).strip()
    if not control:
        return ""

    hints: list[str] = []

    if _has_any(control, ("男性", "男声", "男の声", "男性声", "男っぽい")):
        hints.extend(
            [
                "adult Japanese male voice",
                "mature male timbre",
                "low to medium-low pitch",
            ]
        )
    elif _has_any(control, ("女性", "女声", "女の声", "女性声", "女っぽい")):
        hints.extend(["adult Japanese female voice", "female timbre"])

    if _has_any(control, ("ナレーション", "ナレーター", "アナウンス", "朗読")):
        hints.append("professional narrator voice")
    if _has_any(control, ("落ち着", "穏やか", "冷静", "しっとり")):
        hints.append("calm and composed tone")
    if _has_any(control, ("聞き取りやす", "明瞭", "はっきり", "滑舌")):
        hints.append("clear articulation")
    if _has_any(control, ("ゆっくり", "ゆったり", "少し遅", "遅め")):
        hints.append("slightly slow speaking pace")
    if _has_any(control, ("低い", "低め", "低音", "渋い")):
        hints.append("low pitch")
    if _has_any(control, ("かわいい", "可愛い", "幼い", "少女", "若い女性")):
        hints.append("cute youthful voice")

    if hints:
        return f"{', '.join(dict.fromkeys(hints))}. {control}"
    return control


def _build_language_prompt(language_label: str) -> str:
    language_name = _LANGUAGE_HINTS.get(language_label or "", "")
    if not language_name:
        return ""
    return f"Speak the target text in {language_name}."


def _build_intonation_prompt(intonation_instruction: str) -> str:
    intonation = (intonation_instruction or "").strip()
    intonation = re.sub(r"[()（）]", "", intonation).strip()
    if not intonation:
        return ""

    hints: list[str] = []
    if _has_any(intonation, ("語尾を上げ", "語尾上げ", "上げ調子", "上昇調", "疑問")):
        hints.append("use rising intonation at sentence endings where appropriate")
    if _has_any(intonation, ("語尾を下げ", "文末は下げ", "文末を下げ", "下降調", "落として", "締める")):
        hints.append("use falling intonation at sentence endings")
    if _has_any(intonation, ("平坦", "淡々", "抑揚を抑", "フラット")):
        hints.append("keep the intonation relatively flat and restrained")
    if _has_any(intonation, ("抑揚", "メリハリ", "表情豊か", "感情")):
        hints.append("use expressive pitch variation")
    if _has_any(intonation, ("強調", "はっきり", "際立", "立てる")):
        hints.append("slightly emphasize important words")
    if _has_any(intonation, ("高め", "高く", "明る", "軽やか")):
        hints.append("use a slightly brighter and higher pitch contour")
    if _has_any(intonation, ("低め", "低く", "落ち着", "重く")):
        hints.append("use a slightly lower and calmer pitch contour")
    if _has_any(intonation, ("間", "ポーズ", "ため", "ゆっくり")):
        hints.append("respect pauses and pacing cues in the text")

    if hints:
        return f"Intonation guidance: {', '.join(dict.fromkeys(hints))}. {intonation}"
    return f"Intonation guidance: {intonation}"


def _build_word_accent_prompt(word_accent_instruction: str) -> str:
    raw = (word_accent_instruction or "").strip()
    raw = re.sub(r"[()（）]", "", raw).strip()
    if not raw:
        return ""

    accent_map = {
        "平坦": "flat pitch accent",
        "平板": "flat pitch accent",
        "フラット": "flat pitch accent",
        "語尾上げ": "rising pitch on the final mora",
        "語尾を上げる": "rising pitch on the final mora",
        "上げ": "rising pitch",
        "頭高": "initial high pitch accent with an early drop",
        "頭高型": "initial high pitch accent with an early drop",
        "中高": "middle high pitch accent with a drop after the accented mora",
        "中高型": "middle high pitch accent with a drop after the accented mora",
        "尾高": "final high pitch accent with a drop after the word",
        "尾高型": "final high pitch accent with a drop after the word",
        "低高": "low-to-high pitch pattern",
        "高低": "high-to-low pitch pattern",
    }

    guidance: list[str] = []
    for line in raw.splitlines():
        line = line.strip().strip("・- ")
        if not line:
            continue
        if "=" in line:
            word, accent = line.split("=", 1)
        elif "：" in line:
            word, accent = line.split("：", 1)
        elif ":" in line:
            word, accent = line.split(":", 1)
        else:
            guidance.append(line)
            continue
        word = word.strip(" 「」『』\"'")
        accent = accent.strip()
        if not word or not accent:
            continue
        accent_hint = accent
        for key, value in accent_map.items():
            if key in accent:
                accent_hint = value
                break
        guidance.append(f"pronounce '{word}' with {accent_hint}")

    if not guidance:
        return ""
    return "Word-level pitch accent guidance: " + "; ".join(guidance) + "."


def _prepare_prompt_text_for_continuation(prompt_text: str, target_language: str = "") -> str:
    """Keep prompt transcript as spoken text and add a clean boundary before target text."""
    prompt = (prompt_text or "").strip()
    prompt = re.sub(r"<\|[^>]+?\|>", "", prompt).strip()
    prompt = re.sub(r"\s+", " ", prompt)
    if not prompt:
        return ""

    if prompt[-1] not in "。.!?！？…、,;；:：":
        language_name = _LANGUAGE_HINTS.get(target_language or "", "")
        prompt += "。" if language_name == "Japanese" or _has_any(prompt, ("。", "、")) else "."
    return prompt + " "


class VoxCPMDemo:
    def __init__(
        self,
        model_id: str = "openbmb/VoxCPM2",
        device: str = "auto",
        load_denoiser: bool = True,
    ) -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")
        self.load_denoiser = load_denoiser

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_model_cache_dir = (
            Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic" / "SenseVoiceSmall"
        )
        self.asr_device = "cpu"
        self.asr_model: Optional[AutoModel] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            load_denoiser=self.load_denoiser,
            optimize=self.optimize,
            device=self.device,
        )
        logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        if not self._asr_model_cache_ready():
            raise RuntimeError(
                "自動文字起こしモデル SenseVoiceSmall が未ダウンロード、または途中で止まっています。"
                "参照音声の文字起こし欄に、事前に文字起こしした内容を貼り付けてください。"
            )
        logger.info(
            f"Loading ASR model: {self.asr_model_cache_dir} on device: {self.asr_device}"
        )
        self.asr_model = AutoModel(
            model=str(self.asr_model_cache_dir),
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def _asr_model_cache_ready(self) -> bool:
        if not self.asr_model_cache_dir.exists():
            return False
        required_files = ("config.yaml", "model.pt")
        return all((self.asr_model_cache_dir / filename).exists() for filename in required_files)

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def irodori_project_dir(self) -> Path:
        return Path.cwd() / "external" / "Irodori-TTS"

    @staticmethod
    def _python_executable_works(path: Path) -> bool:
        try:
            result = subprocess.run(
                [str(path), "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except Exception:
            return False

    def irodori_python_path(self) -> Path:
        project_dir = self.irodori_project_dir()
        if sys.platform.startswith("win"):
            candidate = project_dir / ".venv" / "Scripts" / "python.exe"
        else:
            candidate = project_dir / ".venv" / "bin" / "python"
        if candidate.exists() and self._python_executable_works(candidate):
            return candidate
        return Path(sys.executable)

    def irodori_status(self) -> str:
        project_dir = self.irodori_project_dir()
        if not project_dir.exists():
            return "Irodori-TTSは未セットアップです。`scripts\\setup_irodori_tts.ps1` を実行してください。"
        if not (project_dir / "infer.py").exists():
            return f"Irodori-TTSの `infer.py` が見つかりません: {project_dir}"
        if shutil.which("uv") is None:
            return "`uv` コマンドが見つかりません。Irodori-TTSの実行にはuvが必要です。"
        return (
            "Irodori-TTSを使用します。日本語に特化した音声生成・参照音声クローンに対応しています。"
            "多言語発話、VoxCPM2の高精度クローン、自由文による細かな声の指示は未対応です。"
            f"\n\nセットアップ済み: `{project_dir}`"
        )

    def generate_irodori_audio(
        self,
        *,
        text_input: str,
        output_wav_path: str,
        reference_wav_path_input: Optional[str] = None,
        caption_input: str = "",
        lora_adapter_path_input: str = "",
    ) -> Tuple[int, np.ndarray]:
        text = (text_input or "").strip()
        if not text:
            raise ValueError("読み上げテキストを入力してください。")

        project_dir = self.irodori_project_dir()
        infer_py = project_dir / "infer.py"
        if not infer_py.exists():
            raise RuntimeError(
                "Irodori-TTSがまだセットアップされていません。"
                "PowerShellで `scripts\\setup_irodori_tts.ps1` を実行してから、もう一度試してください。"
            )
        python_path = self.irodori_python_path()

        output_path = Path(output_wav_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(python_path),
            str(Path.cwd() / "scripts" / "run_irodori_infer.py"),
            "--hf-checkpoint",
            "Aratako/Irodori-TTS-500M-v3",
            "--text",
            text,
            "--output-wav",
            str(output_path),
        ]
        caption = (caption_input or "").strip()
        if caption:
            command.extend(["--caption", caption])
        if reference_wav_path_input:
            command.extend(["--ref-wav", str(reference_wav_path_input)])
        else:
            command.append("--no-ref")
        lora_adapter = (lora_adapter_path_input or "").strip()
        if lora_adapter:
            command.extend(["--lora-adapter", lora_adapter])

        logger.info("Running Irodori-TTS inference...")
        env = os.environ.copy()
        env.update(
            {
                "UV_NATIVE_TLS": "true",
                "GIT_SSL_BACKEND": "schannel",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.sslBackend",
                "GIT_CONFIG_VALUE_0": "schannel",
            }
        )
        result = subprocess.run(
            command,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        if result.returncode != 0:
            detail = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
            raise RuntimeError(f"Irodori-TTSの生成に失敗しました。\n{detail[-1200:]}")
        if not output_path.exists():
            raise RuntimeError("Irodori-TTSの生成は完了しましたが、出力WAVが見つかりませんでした。")
        return processing_utils.audio_from_file(str(output_path))

    def qwen3_status(self) -> str:
        try:
            import importlib.util

            has_qwen = importlib.util.find_spec("qwen_tts") is not None
        except Exception:
            has_qwen = False

        wrapper_path = Path.cwd() / "scripts" / "run_qwen3_tts_infer.py"
        if not wrapper_path.exists():
            return f"Qwen3-TTS実行ラッパーが見つかりません: {wrapper_path}"
        if not has_qwen:
            return (
                "VoiceDesignCloner連携（Qwen3-TTS・簡易）は未セットアップです。PowerShellで "
                "`scripts\\setup_qwen3_tts.ps1` を実行してください。"
            )
        return (
            "VoiceDesignCloner連携（Qwen3-TTS・簡易）を使用します。Voice-Design-ClonerのQwen3-TTSワークフローを参考に、"
            "多言語の声デザイン、生成数指定による複数候補、参照音声+文字起こしによる簡易クローン、"
            "選んだ声での簡易コーパス一括音声化、リサンプル、esd.list生成、Irodori-TTS LoRA学習データ準備、"
            "LoRA学習実行入口に対応しています。Style-Bert-VITS2向けの完全自動配置はまだ統合していません。"
        )

    def generate_qwen3_audio(
        self,
        *,
        mode: str,
        text_input: str,
        output_wav_path: str,
        language_input: str,
        instruct_input: str = "",
        reference_wav_path_input: Optional[str] = None,
        reference_text_input: str = "",
    ) -> Tuple[int, np.ndarray]:
        text = (text_input or "").strip()
        if not text:
            raise ValueError("読み上げテキストを入力してください。")
        if mode == "clone" and not (reference_text_input or "").strip():
            raise ValueError("Qwen3-TTSの声のクローンには、参照音声の文字起こしが必要です。")

        wrapper_path = Path.cwd() / "scripts" / "run_qwen3_tts_infer.py"
        if not wrapper_path.exists():
            raise RuntimeError(f"Qwen3-TTS実行ラッパーが見つかりません: {wrapper_path}")

        output_path = Path(output_wav_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(wrapper_path),
            "--mode",
            mode,
            "--text",
            text,
            "--output-wav",
            str(output_path),
            "--language",
            language_input,
        ]
        instruct = (instruct_input or "").strip()
        if instruct:
            command.extend(["--instruct", instruct])
        if reference_wav_path_input:
            command.extend(["--ref-wav", str(reference_wav_path_input)])
        ref_text = (reference_text_input or "").strip()
        if ref_text:
            command.extend(["--ref-text", ref_text])

        logger.info("Running Qwen3-TTS inference...")
        env = os.environ.copy()
        env.update(
            {
                "UV_NATIVE_TLS": "true",
                "GIT_SSL_BACKEND": "schannel",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.sslBackend",
                "GIT_CONFIG_VALUE_0": "schannel",
            }
        )
        result = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        if result.returncode != 0:
            detail = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
            raise RuntimeError(f"Qwen3-TTSの生成に失敗しました。\n{detail[-1200:]}")
        if not output_path.exists():
            raise RuntimeError("Qwen3-TTSの生成は完了しましたが、出力WAVが見つかりませんでした。")
        return processing_utils.audio_from_file(str(output_path))

    def generate_qwen3_corpus(
        self,
        *,
        texts_file_path: str,
        output_dir_path: str,
        language_input: str,
        reference_wav_path_input: str,
        reference_text_input: str,
        target_sr: int = 44100,
    ) -> tuple[str, str, str]:
        if not reference_wav_path_input:
            raise ValueError("コーパス一括音声化には参照音声が必要です。")
        if not (reference_text_input or "").strip():
            raise ValueError("コーパス一括音声化には参照音声の文字起こしが必要です。")

        wrapper_path = Path.cwd() / "scripts" / "run_qwen3_tts_infer.py"
        if not wrapper_path.exists():
            raise RuntimeError(f"Qwen3-TTS実行ラッパーが見つかりません: {wrapper_path}")

        output_dir = Path(output_dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(wrapper_path),
            "--mode",
            "clone-batch",
            "--texts-file",
            str(texts_file_path),
            "--output-dir",
            str(output_dir),
            "--text-list",
            "Neutral.txt",
            "--target-sr",
            str(int(target_sr or 0)),
            "--language",
            language_input,
            "--ref-wav",
            str(reference_wav_path_input),
            "--ref-text",
            (reference_text_input or "").strip(),
        ]

        logger.info("Running Qwen3-TTS corpus batch...")
        env = os.environ.copy()
        env.update(
            {
                "UV_NATIVE_TLS": "true",
                "GIT_SSL_BACKEND": "schannel",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.sslBackend",
                "GIT_CONFIG_VALUE_0": "schannel",
            }
        )
        result = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
        )
        if result.returncode != 0:
            detail = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
            raise RuntimeError(f"Qwen3-TTSのコーパス一括音声化に失敗しました。\n{detail[-2000:]}")

        raw_dir = output_dir / "raw"
        text_list = output_dir / "Neutral.txt"
        if not raw_dir.exists() or not text_list.exists():
            raise RuntimeError("コーパス生成は完了しましたが、rawフォルダまたはNeutral.txtが見つかりませんでした。")
        return str(output_dir), str(raw_dir), str(text_list)

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        intonation_instruction: str = "",
        word_accent_instruction: str = "",
        target_language: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
    ) -> Tuple[int, np.ndarray]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("読み上げテキストを入力してください。")

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = _prepare_prompt_text_for_continuation(prompt_text, target_language) or None

        if prompt_text_clean:
            # Continuation / high-fidelity cloning treats the target text as literal spoken text.
            # Do not prepend English control prompts here, or they may be spoken at the start.
            control = ""
            final_text = text
        else:
            language_prompt = _build_language_prompt(target_language)
            control_parts = [
                part
                for part in (
                    language_prompt,
                    _build_control_prompt(control_instruction),
                    _build_intonation_prompt(intonation_instruction),
                    _build_word_accent_prompt(word_accent_instruction),
                )
                if part
            ]
            control = " ".join(control_parts)
            final_text = f"({control}){text}" if control else text

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
        )
        wav = current_model.generate(**generate_kwargs)
        return (current_model.tts_model.sample_rate, wav)


# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])
    settings_path = Path.cwd() / ".jpvoxcpm_settings.json"
    initial_output_dir = (Path.cwd() / "outputs").resolve()
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        saved_output_dir = (settings.get("output_dir") or "").strip()
        if saved_output_dir:
            expanded = os.path.expandvars(os.path.expanduser(saved_output_dir))
            saved_path = Path(expanded)
            if not saved_path.is_absolute():
                saved_path = Path.cwd() / saved_path
            initial_output_dir = saved_path.resolve()
    except Exception:
        pass
    current_output_dir = {"path": initial_output_dir}

    def _list_voice_design_history():
        output_dir = _output_dir(create=False)
        if not output_dir.exists():
            return []
        history_paths = []
        for pattern in (
            "voice_design_*.wav",
            "irodori_design_*.wav",
            "irodori_history_reuse_*.wav",
            "qwen3_design_*.wav",
            "qwen3_gacha_*.wav",
            "qwen3_history_reuse_*.wav",
        ):
            history_paths.extend(output_dir.glob(pattern))
        choices = []
        for path in sorted(set(history_paths), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%m/%d %H:%M")
            choices.append((f"{timestamp} - {path.name}", str(path)))
        return choices

    def _resolve_output_dir(folder_path: str = "") -> Path:
        folder_path = (folder_path or "").strip()
        if not folder_path:
            return (Path.cwd() / "outputs").resolve()
        expanded = os.path.expandvars(os.path.expanduser(folder_path))
        output_dir = Path(expanded)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        return output_dir.resolve()

    def _output_dir(create: bool = True) -> Path:
        output_dir = current_output_dir["path"]
        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _history_dropdown_update(default_to_first: bool = True):
        choices = _list_voice_design_history()
        value = choices[0][1] if default_to_first and choices else None
        return gr.update(choices=choices, value=value)

    def _set_output_dir(folder_path: str):
        try:
            output_dir = _resolve_output_dir(folder_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            if not output_dir.is_dir():
                raise ValueError("指定されたパスはフォルダではありません。")
            current_output_dir["path"] = output_dir
            folder_text = str(output_dir)
            settings_message = ""
            try:
                settings_path.write_text(
                    json.dumps({"output_dir": folder_text}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as settings_error:
                settings_message = f"\n\n設定ファイルへの保存はできませんでした: {settings_error}"
            history_update = _history_dropdown_update()
            message = f"保存先フォルダを変更しました: {folder_text}{settings_message}"
            return (
                gr.update(value=folder_text),
                gr.update(value=folder_text),
                gr.update(value=folder_text),
                gr.update(value=folder_text),
                history_update,
                history_update,
                history_update,
                message,
            )
        except Exception as e:
            folder_text = str(_output_dir(create=False))
            message = f"保存先フォルダを変更できませんでした: {e}"
            return (
                gr.update(value=folder_text),
                gr.update(value=folder_text),
                gr.update(value=folder_text),
                gr.update(value=folder_text),
                gr.update(),
                gr.update(),
                gr.update(),
                message,
            )

    def _open_output_dir(folder_path: str):
        try:
            output_dir = _resolve_output_dir(folder_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            if not output_dir.is_dir():
                raise ValueError("指定されたパスはフォルダではありません。")
            if sys.platform.startswith("win"):
                os.startfile(str(output_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(output_dir)])
            else:
                subprocess.Popen(["xdg-open", str(output_dir)])
            return f"保存先フォルダを開きました: {output_dir}"
        except Exception as e:
            return f"保存先フォルダを開けませんでした: {e}"

    def _open_existing_folder(folder_path: str):
        try:
            if not (folder_path or "").strip():
                raise ValueError("先にコーパスを生成してください。")
            folder = Path(folder_path or "").expanduser().resolve()
            if not folder.exists() or not folder.is_dir():
                raise ValueError("フォルダが見つかりません。")
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
            return f"フォルダを開きました: {folder}"
        except Exception as e:
            return f"フォルダを開けませんでした: {e}"

    def _sanitize_filename(name: str) -> str:
        name = (name or "").strip()
        if not name:
            return ""
        name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
        name = re.sub(r"\s+", "_", name).strip("._ ")
        return name[:80]

    def _read_corpus_file_text(corpus_file: Optional[str]) -> str:
        if not corpus_file:
            return ""
        try:
            file_path = corpus_file if isinstance(corpus_file, str) else getattr(corpus_file, "name", "")
            if not file_path:
                return ""
            return Path(file_path).read_text(encoding="utf-8-sig")
        except Exception as e:
            raise RuntimeError(f"コーパスTXTを読み込めませんでした: {e}") from e

    def _collect_corpus_lines(corpus_text: str, corpus_file: Optional[str], max_lines: int) -> list[str]:
        combined_text = "\n".join(
            part for part in ((corpus_text or "").strip(), _read_corpus_file_text(corpus_file).strip()) if part
        )
        lines = [line.strip() for line in combined_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("コーパス本文を入力するか、TXTファイルをアップロードしてください。")
        limit = int(max_lines or 0)
        if limit > 0:
            lines = lines[:limit]
        return lines

    def _resolve_corpus_output_folder(folder_path: str) -> Path:
        if not (folder_path or "").strip():
            raise ValueError("コーパスフォルダを指定してください。")
        folder = Path(os.path.expandvars(os.path.expanduser(folder_path.strip())))
        if not folder.is_absolute():
            folder = _output_dir() / folder
        folder = folder.resolve()
        output_root = _output_dir().resolve()
        if folder != output_root and output_root not in folder.parents:
            raise ValueError("現在の保存先フォルダ内のコーパスフォルダだけ処理できます。")
        if not folder.is_dir():
            raise ValueError(f"コーパスフォルダが見つかりません: {folder}")
        return folder

    def _load_corpus_text_map(base_dir: Path) -> dict[str, str]:
        text_file = base_dir / "Neutral.txt"
        if not text_file.exists():
            raise ValueError(f"Neutral.txt が見つかりません: {text_file}")
        text_map: dict[str, str] = {}
        lines = [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, line in enumerate(lines):
            parts = line.split("|")
            if len(parts) >= 4:
                key = parts[0].strip()
                text = parts[3].strip()
            elif len(parts) == 2:
                key = parts[0].strip()
                text = parts[1].strip()
            else:
                key = f"{index + 1:04d}"
                text = line
            if not key or not text:
                continue
            stem = Path(key).stem
            text_map[stem] = text
            text_map[f"{stem}.wav"] = text
        if not text_map:
            raise ValueError("Neutral.txt に利用できる本文がありません。")
        return text_map

    def _lora_lab_root() -> Path:
        root = _output_dir() / "lora_data" / "lab"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _resample_corpus_raw(folder_path: str, target_sr: int, progress=gr.Progress()):
        base_dir = _resolve_corpus_output_folder(folder_path)
        raw_dir = base_dir / "raw"
        if not raw_dir.is_dir():
            raise ValueError(f"rawフォルダが見つかりません: {raw_dir}")
        wav_files = sorted(raw_dir.glob("*.wav"))
        if not wav_files:
            raise ValueError("rawフォルダにWAVがありません。")

        import librosa
        import soundfile as sf

        sr = int(target_sr or 44100)
        out_dir = base_dir / "resampled"
        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        for index, wav_path in enumerate(wav_files):
            audio, source_sr = librosa.load(str(wav_path), sr=None, mono=True)
            if int(source_sr) != sr:
                audio = librosa.resample(audio, orig_sr=int(source_sr), target_sr=sr)
            sf.write(str(out_dir / wav_path.name), audio, sr, subtype="PCM_16")
            progress((index + 1) / len(wav_files), f"{index + 1}/{len(wav_files)}")
        elapsed = time.time() - start
        return f"{len(wav_files)}ファイルを {sr}Hz / mono / PCM_16 に変換しました（{elapsed:.1f}秒）。\n出力先: {out_dir}"

    def _generate_corpus_esd_list(folder_path: str, speaker_name: str, lang_code: str):
        base_dir = _resolve_corpus_output_folder(folder_path)
        raw_dir = base_dir / "raw"
        if not raw_dir.is_dir():
            raise ValueError(f"rawフォルダが見つかりません: {raw_dir}")
        wav_files = sorted(raw_dir.glob("*.wav"))
        if not wav_files:
            raise ValueError("rawフォルダにWAVがありません。")

        text_map = _load_corpus_text_map(base_dir)
        speaker = _sanitize_filename(speaker_name) or _sanitize_filename(base_dir.name) or "speaker"
        lang = (lang_code or "JP").strip().upper()
        esd_lines = []
        skipped = []
        for wav_path in wav_files:
            text = text_map.get(wav_path.name) or text_map.get(wav_path.stem)
            if not text:
                skipped.append(wav_path.name)
                continue
            esd_lines.append(f"{wav_path.name}|{speaker}|{lang}|{text}")
        if not esd_lines:
            raise ValueError("raw/*.wav と Neutral.txt の対応が見つかりませんでした。")

        esd_path = base_dir / "esd.list"
        esd_path.write_text("\n".join(esd_lines) + "\n", encoding="utf-8")
        message = f"{len(esd_lines)}行の esd.list を生成しました。\n保存先: {esd_path}"
        if skipped:
            message += f"\n本文が見つからずスキップ: {len(skipped)}件"
        return message, str(esd_path)

    def _prepare_irodori_lora_data(
        folder_path: str,
        speaker_name: str,
        emotion_name: str,
        wav_folder_name: str,
    ):
        base_dir = _resolve_corpus_output_folder(folder_path)
        text_map = _load_corpus_text_map(base_dir)
        wav_folder = (wav_folder_name or "raw").strip()
        if wav_folder not in {"raw", "resampled"}:
            raise ValueError("WAVフォルダは raw または resampled を選んでください。")
        source_wav_dir = base_dir / wav_folder
        if not source_wav_dir.is_dir():
            raise ValueError(f"{wav_folder}フォルダが見つかりません: {source_wav_dir}")
        wav_files = sorted(source_wav_dir.glob("*.wav"))
        if not wav_files:
            raise ValueError(f"{wav_folder}フォルダにWAVがありません。")

        speaker = _sanitize_filename(speaker_name) or _sanitize_filename(base_dir.name) or "speaker"
        emotion = _sanitize_filename(emotion_name) or "Neutral"
        lab_root = _lora_lab_root()
        dest_dir = lab_root / speaker / emotion
        dest_wavs = dest_dir / "wavs"
        resolved_lab_root = lab_root.resolve()
        resolved_dest_wavs = dest_wavs.resolve()
        if resolved_lab_root not in resolved_dest_wavs.parents:
            raise ValueError("LoRA学習データの出力先を安全に解決できませんでした。")
        if dest_wavs.exists():
            shutil.rmtree(dest_wavs)
        dest_wavs.mkdir(parents=True, exist_ok=True)

        txt_lines = []
        jsonl_rows = []
        skipped = []
        for wav_path in wav_files:
            text = text_map.get(wav_path.name) or text_map.get(wav_path.stem)
            if not text:
                skipped.append(wav_path.name)
                continue
            dest_wav = dest_wavs / wav_path.name
            shutil.copy2(wav_path, dest_wav)
            txt_lines.append(f"{wav_path.stem}: {text}")
            jsonl_rows.append(
                json.dumps(
                    {"audio": str(dest_wav.resolve()), "text": text},
                    ensure_ascii=False,
                )
            )

        if not txt_lines:
            raise ValueError(f"{wav_folder}/*.wav と Neutral.txt の対応が見つかりませんでした。")

        dest_dir.mkdir(parents=True, exist_ok=True)
        lab_text_path = dest_dir / f"{emotion}.txt"
        lab_text_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
        jsonl_path = _output_dir() / "lora_data" / f"{speaker}_{emotion}.jsonl"
        jsonl_path.write_text("\n".join(jsonl_rows) + "\n", encoding="utf-8")

        status = (
            f"{len(txt_lines)}件をIrodori-TTS LoRA学習用データに変換しました。\n\n"
            f"- lab: `{dest_dir}`\n"
            f"- wavs: `{dest_wavs}`\n"
            f"- text: `{lab_text_path}`\n"
            f"- jsonl: `{jsonl_path}`"
        )
        if skipped:
            status += f"\n\n本文が見つからずスキップ: {len(skipped)}件"
        return status, str(dest_dir), str(dest_dir), str(lab_text_path), str(jsonl_path)

    def _resolve_lora_lab_dir(lab_dir_path: str) -> Path:
        if not (lab_dir_path or "").strip():
            raise ValueError("学習するlabフォルダを指定してください。")
        lab_dir = Path(os.path.expandvars(os.path.expanduser(lab_dir_path.strip())))
        if not lab_dir.is_absolute():
            lab_dir = _lora_lab_root() / lab_dir
        lab_dir = lab_dir.resolve()
        lab_root = _lora_lab_root().resolve()
        if lab_root not in lab_dir.parents:
            raise ValueError("現在の保存先フォルダ内の lora_data/lab 配下だけ学習できます。")
        if not lab_dir.is_dir():
            raise ValueError(f"labフォルダが見つかりません: {lab_dir}")
        return lab_dir

    def _write_lora_training_jsonl_from_lab(lab_dir: Path) -> tuple[str, str, Path]:
        speaker = _sanitize_filename(lab_dir.parent.name) or "speaker"
        emotion = _sanitize_filename(lab_dir.name) or "Neutral"
        text_file = lab_dir / f"{emotion}.txt"
        wav_dir = lab_dir / "wavs"
        if not text_file.is_file():
            raise ValueError(f"labテキストが見つかりません: {text_file}")
        if not wav_dir.is_dir():
            raise ValueError(f"wavsフォルダが見つかりません: {wav_dir}")

        rows = []
        for line in text_file.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if not clean or ":" not in clean:
                continue
            file_id, text = clean.split(":", 1)
            stem = file_id.strip()
            wav_path = wav_dir / f"{stem}.wav"
            if wav_path.is_file():
                rows.append(json.dumps({"audio": str(wav_path.resolve()), "text": text.strip()}, ensure_ascii=False))
        if not rows:
            raise ValueError("labテキストとWAVの対応が見つかりませんでした。")

        jsonl_dir = _output_dir() / "lora_data" / "jsonl"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_dir / f"{speaker}_{emotion}.jsonl"
        jsonl_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return speaker, emotion, jsonl_path

    def _check_irodori_lora_lab_data(lab_dir_path: str) -> str:
        lab_dir = _resolve_lora_lab_dir(lab_dir_path)
        speaker = _sanitize_filename(lab_dir.parent.name) or "speaker"
        emotion = _sanitize_filename(lab_dir.name) or "Neutral"
        text_file = lab_dir / f"{emotion}.txt"
        wav_dir = lab_dir / "wavs"
        if not text_file.is_file():
            raise ValueError(f"labテキストが見つかりません: {text_file}")
        if not wav_dir.is_dir():
            raise ValueError(f"wavsフォルダが見つかりません: {wav_dir}")

        text_entries: dict[str, str] = {}
        malformed_lines = 0
        empty_text_lines = 0
        for line in text_file.read_text(encoding="utf-8-sig").splitlines():
            clean = line.strip()
            if not clean:
                continue
            if ":" not in clean:
                malformed_lines += 1
                continue
            file_id, text = clean.split(":", 1)
            stem = Path(file_id.strip()).stem
            body = text.strip()
            if not stem or not body:
                empty_text_lines += 1
                continue
            text_entries[stem] = body

        wav_files = sorted(wav_dir.glob("*.wav"))
        wav_by_stem = {path.stem: path for path in wav_files}
        matched = sorted(stem for stem in text_entries if stem in wav_by_stem)
        missing_wavs = sorted(stem for stem in text_entries if stem not in wav_by_stem)
        extra_wavs = sorted(stem for stem in wav_by_stem if stem not in text_entries)

        durations: list[float] = []
        sample_rates: dict[int, int] = {}
        channels: dict[int, int] = {}
        subtypes: dict[str, int] = {}
        short_files: list[str] = []
        long_files: list[str] = []
        unreadable_files: list[str] = []
        for stem in matched:
            wav_path = wav_by_stem[stem]
            try:
                import soundfile as sf

                info = sf.info(str(wav_path))
                duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
                durations.append(duration)
                sample_rates[int(info.samplerate)] = sample_rates.get(int(info.samplerate), 0) + 1
                channels[int(info.channels)] = channels.get(int(info.channels), 0) + 1
                subtype = str(info.subtype or "unknown")
                subtypes[subtype] = subtypes.get(subtype, 0) + 1
                if duration < 0.7:
                    short_files.append(wav_path.name)
                if duration > 30.0:
                    long_files.append(wav_path.name)
            except Exception:
                unreadable_files.append(wav_path.name)

        warnings: list[str] = []
        if len(matched) < 10:
            warnings.append("学習データが10件未満です。動作確認はできますが、声質学習にはかなり少なめです。")
        elif len(matched) < 50:
            warnings.append("声質として使うには少なめです。まず動作確認し、可能なら50件以上へ増やしてください。")
        if missing_wavs:
            warnings.append(f"テキストはあるがWAVがない項目があります: {len(missing_wavs)}件")
        if extra_wavs:
            warnings.append(f"WAVはあるがテキストにない項目があります: {len(extra_wavs)}件")
        if malformed_lines or empty_text_lines:
            warnings.append(f"読み取れない行があります: 形式不正 {malformed_lines}件 / 空テキスト {empty_text_lines}件")
        if short_files:
            warnings.append(f"0.7秒未満の短いWAVがあります: {len(short_files)}件")
        if long_files:
            warnings.append(f"30秒を超える長いWAVがあります: {len(long_files)}件")
        if unreadable_files:
            warnings.append(f"読み込めないWAVがあります: {len(unreadable_files)}件")
        if any(ch != 1 for ch in channels):
            warnings.append("モノラル以外のWAVがあります。可能ならリサンプルで mono / PCM_16 に整えてください。")

        total_sec = sum(durations)
        avg_sec = (total_sec / len(durations)) if durations else 0.0
        min_sec = min(durations) if durations else 0.0
        max_sec = max(durations) if durations else 0.0
        status = "OK" if not warnings else "確認してください"
        lines = [
            f"品質チェック: {status}",
            "",
            f"- 話者: {speaker}",
            f"- 感情: {emotion}",
            f"- lab: {lab_dir}",
            f"- 対応済みデータ: {len(matched)}件",
            f"- テキスト行: {len(text_entries)}件",
            f"- WAV: {len(wav_files)}件",
            f"- 合計時間: {total_sec:.1f}秒",
            f"- 長さ: 平均 {avg_sec:.2f}秒 / 最短 {min_sec:.2f}秒 / 最長 {max_sec:.2f}秒",
            f"- サンプルレート: {', '.join(f'{sr}Hz x{count}' for sr, count in sorted(sample_rates.items())) or '未確認'}",
            f"- チャンネル: {', '.join(f'{ch}ch x{count}' for ch, count in sorted(channels.items())) or '未確認'}",
            f"- 形式: {', '.join(f'{name} x{count}' for name, count in sorted(subtypes.items())) or '未確認'}",
        ]
        if warnings:
            lines.extend(["", "注意:", *[f"- {warning}" for warning in warnings]])
        else:
            lines.extend(["", "このままドライランへ進めます。実学習は短いステップ数から試してください。"])

        def sample(names: list[str]) -> str:
            head = names[:5]
            suffix = "" if len(names) <= 5 else f" ほか{len(names) - 5}件"
            return ", ".join(head) + suffix

        details = [
            ("WAV不足", missing_wavs),
            ("テキストなしWAV", extra_wavs),
            ("短すぎるWAV", short_files),
            ("長すぎるWAV", long_files),
            ("読み込み不可WAV", unreadable_files),
        ]
        for label, names in details:
            if names:
                lines.append(f"{label}: {sample(names)}")
        return "\n".join(lines)

    def _irodori_python_path() -> Path:
        return demo.irodori_python_path()

    def _lora_training_paths(speaker: str) -> tuple[Path, Path, Path]:
        base_dir = _output_dir() / "lora_data"
        latent_dir = base_dir / "latents" / speaker
        manifest_path = base_dir / "manifests" / f"{speaker}_manifest.jsonl"
        output_dir = _output_dir() / "lora" / speaker
        return latent_dir, manifest_path, output_dir

    def _is_lora_adapter_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        has_config = (path / "adapter_config.json").is_file()
        has_weights = (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
        return has_config and has_weights

    def _resolve_lora_adapter_dir(folder: Path) -> Optional[Path]:
        if _is_lora_adapter_dir(folder):
            return folder
        candidates = [p for p in folder.rglob("adapter_config.json") if _is_lora_adapter_dir(p.parent)]
        if not candidates:
            return None

        def sort_key(config_path: Path):
            parts = "/".join(config_path.parts).lower()
            final_rank = 0 if ("final" in parts or "last" in parts) else 1
            return final_rank, -config_path.stat().st_mtime

        return sorted(candidates, key=sort_key)[0].parent

    _LORA_STUDIO_METADATA = "jp_voice_studio_metadata.json"

    def _read_lora_studio_metadata(adapter_dir: Path) -> dict:
        for metadata_path in (adapter_dir / _LORA_STUDIO_METADATA, adapter_dir.parent / _LORA_STUDIO_METADATA):
            if not metadata_path.is_file():
                continue
            try:
                return json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
        return {}

    def _count_jsonl_rows(path: Path) -> int:
        if not path.is_file():
            return 0
        try:
            return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        except Exception:
            return 0

    def _write_lora_studio_metadata(
        output_dir: Path,
        adapter_dir: Path,
        *,
        speaker: str,
        emotion: str,
        lab_dir: Path,
        jsonl_path: Path,
        manifest_path: Path,
        init_checkpoint: str,
        max_steps: int,
        batch_size: int,
        num_workers: int,
        learning_rate: float,
    ) -> None:
        metadata = {
            "schema_version": 1,
            "display_name": speaker,
            "note": "",
            "engine": "Irodori-TTS",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "speaker": speaker,
            "emotion": emotion,
            "sample_count": _count_jsonl_rows(jsonl_path),
            "manifest_count": _count_jsonl_rows(manifest_path),
            "max_steps": max_steps,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": learning_rate,
            "lab_dir": str(lab_dir),
            "jsonl_path": str(jsonl_path),
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
            "adapter_dir": str(adapter_dir),
            "base_checkpoint": init_checkpoint,
        }
        for target in (output_dir / _LORA_STUDIO_METADATA, adapter_dir / _LORA_STUDIO_METADATA):
            try:
                target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to write LoRA metadata %s: %s", target, e)

    def _lora_adapter_label(speaker_dir: Path, adapter_dir: Path) -> str:
        metadata = _read_lora_studio_metadata(adapter_dir)
        display_name = (metadata.get("display_name") or speaker_dir.name).strip()
        sample_count = metadata.get("sample_count")
        max_steps = metadata.get("max_steps")
        parts = [display_name, adapter_dir.name]
        if sample_count:
            parts.append(f"{sample_count}件")
        if max_steps:
            parts.append(f"{max_steps}step")
        try:
            updated = datetime.fromtimestamp(adapter_dir.stat().st_mtime).strftime("%m/%d %H:%M")
        except OSError:
            updated = ""
        label = " - ".join(str(part) for part in parts if str(part))
        return f"{label} ({updated})" if updated else label

    def _list_irodori_lora_adapters() -> list[tuple[str, str]]:
        lora_root = _output_dir(create=False) / "lora"
        choices: list[tuple[str, str]] = [("使用しない", "")]
        if not lora_root.is_dir():
            return choices
        for speaker_dir in sorted((p for p in lora_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            adapter_dir = _resolve_lora_adapter_dir(speaker_dir)
            if adapter_dir is None:
                continue
            choices.append((_lora_adapter_label(speaker_dir, adapter_dir), str(adapter_dir)))
        return choices

    def _lora_adapter_dropdown_update():
        choices = _list_irodori_lora_adapters()
        value = choices[0][1] if choices else ""
        return gr.update(choices=choices, value=value)

    def _lora_adapter_summary() -> str:
        lora_root = _output_dir(create=False) / "lora"
        if not lora_root.is_dir():
            return f"学習済みLoRAアダプタはまだありません。\n保存先: {lora_root}"

        rows = []
        for speaker_dir in sorted((p for p in lora_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            adapter_dir = _resolve_lora_adapter_dir(speaker_dir)
            if adapter_dir is None:
                rows.append(f"- {speaker_dir.name}: アダプタ未検出")
                continue
            weights = adapter_dir / "adapter_model.safetensors"
            if not weights.is_file():
                weights = adapter_dir / "adapter_model.bin"
            size_mb = weights.stat().st_size / (1024 * 1024) if weights.is_file() else 0.0
            updated = datetime.fromtimestamp(adapter_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            metadata = _read_lora_studio_metadata(adapter_dir)
            display_name = (metadata.get("display_name") or speaker_dir.name).strip()
            sample_count = metadata.get("sample_count")
            sample_text = f"{sample_count}件" if isinstance(sample_count, int) and sample_count > 0 else "不明"
            max_steps = metadata.get("max_steps") or "不明"
            lr_value = metadata.get("learning_rate")
            lr_text = f"{float(lr_value):.2g}" if isinstance(lr_value, (int, float)) else "不明"
            created_at = metadata.get("created_at") or "不明"
            note = (metadata.get("note") or "").strip()
            note_text = f"\n  メモ: {note}" if note else ""
            rows.append(
                f"- {display_name}: {adapter_dir.name} / {size_mb:.1f} MB / 更新 {updated}\n"
                f"  データ: {sample_text} / steps: {max_steps} / lr: {lr_text} / 作成: {created_at}{note_text}\n"
                f"  {adapter_dir}"
            )

        if not rows:
            return f"学習済みLoRAアダプタはまだありません。\n保存先: {lora_root}"
        return "学習済みLoRAアダプタ一覧\n\n" + "\n".join(rows) + f"\n\n保存先: {lora_root}"

    def _refresh_lora_adapter_management():
        choices = _list_irodori_lora_adapters()
        value = choices[0][1] if choices else ""
        return (
            gr.update(choices=choices, value=value),
            gr.update(choices=choices, value=value),
            gr.update(choices=choices, value=value),
            _lora_adapter_summary(),
        )

    def _open_lora_root():
        lora_root = _output_dir() / "lora"
        lora_root.mkdir(parents=True, exist_ok=True)
        return _open_existing_folder(str(lora_root))

    def _run_subprocess_lines(command: list[str], cwd: Path):
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                yield line.rstrip()
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"コマンドが失敗しました（exit={process.returncode}）: {' '.join(command)}")
        except GeneratorExit:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise

    def _run_irodori_lora_training(
        lab_dir_path: str,
        max_steps: int,
        batch_size: int,
        num_workers: int,
        learning_rate: float,
        dry_run: bool,
    ):
        logs: list[str] = []

        def emit(status: str):
            return status, "\n".join(logs[-80:])

        try:
            lab_dir = _resolve_lora_lab_dir(lab_dir_path)
            speaker, emotion, jsonl_path = _write_lora_training_jsonl_from_lab(lab_dir)
            irodori_root = demo.irodori_project_dir()
            irodori_python = _irodori_python_path()
            encode_script = Path.cwd() / "scripts" / "encode_irodori_latents.py"
            train_script = Path.cwd() / "scripts" / "run_irodori_train.py"
            train_config = irodori_root / "configs" / "train_500m_v3_lora.yaml"
            latent_dir, manifest_path, output_dir = _lora_training_paths(speaker)
            max_steps_i = max(1, int(max_steps or 50))
            batch_i = max(1, int(batch_size or 1))
            workers_i = max(0, int(num_workers or 0))
            lr_value = float(learning_rate or 0.0001)

            for required in (irodori_python, encode_script, train_script, train_config):
                if not required.exists():
                    raise RuntimeError(f"必要なファイルが見つかりません: {required}")

            encode_command = [
                str(irodori_python),
                str(encode_script),
                "--input-jsonl",
                str(jsonl_path),
                "--latent-dir",
                str(latent_dir),
                "--manifest",
                str(manifest_path),
            ]
            checkpoint_command = [
                str(irodori_python),
                "-c",
                (
                    "try:\n"
                    "    import truststore\n"
                    "    truststore.inject_into_ssl()\n"
                    "except Exception:\n"
                    "    pass\n"
                    "from huggingface_hub import hf_hub_download; "
                    "print(hf_hub_download('Aratako/Irodori-TTS-500M-v3', 'model.safetensors'))"
                ),
            ]
            train_command_template = [
                str(irodori_python),
                str(train_script),
                "--config",
                str(train_config),
                "--manifest",
                str(manifest_path),
                "--init-checkpoint",
                "<downloaded model.safetensors>",
                "--output-dir",
                str(output_dir),
                "--lora",
                "--max-steps",
                str(max_steps_i),
                "--batch-size",
                str(batch_i),
                "--num-workers",
                str(workers_i),
                "--lr",
                str(lr_value),
            ]

            logs.append(f"[準備] lab={lab_dir}")
            logs.append(f"[準備] jsonl={jsonl_path}")
            logs.append("[コマンド] latent encode:")
            logs.append(" ".join(encode_command))
            logs.append("[コマンド] checkpoint:")
            logs.append(" ".join(checkpoint_command))
            logs.append("[コマンド] train:")
            logs.append(" ".join(train_command_template))
            if dry_run:
                yield emit("ドライラン完了。チェックを外すと実際にlatentエンコードとLoRA学習を実行します。")
                return

            yield emit("latentエンコードを開始します。")
            for line in _run_subprocess_lines(encode_command, irodori_root):
                logs.append(line)
                yield emit("latentエンコード中...")
            if not manifest_path.is_file():
                raise RuntimeError(f"manifestが生成されませんでした: {manifest_path}")

            yield emit("初期チェックポイントを確認しています。")
            checkpoint_lines = list(_run_subprocess_lines(checkpoint_command, irodori_root))
            logs.extend(checkpoint_lines)
            init_checkpoint = checkpoint_lines[-1].strip() if checkpoint_lines else ""
            if not init_checkpoint or not Path(init_checkpoint).is_file():
                raise RuntimeError("初期チェックポイントを取得できませんでした。")

            train_command = train_command_template.copy()
            train_command[train_command.index("<downloaded model.safetensors>")] = init_checkpoint
            output_dir.mkdir(parents=True, exist_ok=True)
            yield emit("LoRA学習を開始します。")
            for line in _run_subprocess_lines(train_command, irodori_root):
                logs.append(line)
                yield emit("LoRA学習中...")
            adapter_dir = _resolve_lora_adapter_dir(output_dir) or output_dir
            _write_lora_studio_metadata(
                output_dir,
                adapter_dir,
                speaker=speaker,
                emotion=emotion,
                lab_dir=lab_dir,
                jsonl_path=jsonl_path,
                manifest_path=manifest_path,
                init_checkpoint=init_checkpoint,
                max_steps=max_steps_i,
                batch_size=batch_i,
                num_workers=workers_i,
                learning_rate=lr_value,
            )
            logs.append(f"[メタ情報] {adapter_dir / _LORA_STUDIO_METADATA}")
            yield emit(f"LoRA学習が完了しました。出力先: {output_dir}")
        except Exception as e:
            logs.append(f"ERROR: {e}")
            yield emit(f"LoRA学習の準備または実行に失敗しました: {e}")

    def _save_wav_for_download(sr: int, wav_np: np.ndarray, prefix: str, filename_hint: str = "") -> str:
        output_dir = _output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_name = _sanitize_filename(filename_hint)
        if custom_name:
            output_path = output_dir / f"{prefix}_{custom_name}.wav"
            if output_path.exists():
                output_path = output_dir / f"{prefix}_{custom_name}_{timestamp}.wav"
        else:
            output_path = output_dir / f"{prefix}_{timestamp}_{uuid4().hex[:8]}.wav"
        processing_utils.audio_to_file(sr, np.asarray(wav_np), str(output_path), format="wav")
        logger.info(f"Saved generated WAV for download: {output_path}")
        return str(output_path)

    def _write_reference_text_sidecar(wav_path: str, text: str) -> None:
        clean_text = (text or "").strip()
        if not clean_text:
            return
        try:
            Path(wav_path).with_suffix(".txt").write_text(clean_text, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write reference text sidecar: %s", e)

    def _read_reference_text_sidecar(wav_path: Optional[str]) -> str:
        if not wav_path:
            return ""
        try:
            sidecar = Path(wav_path).with_suffix(".txt")
            if sidecar.exists():
                return sidecar.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning("Failed to read reference text sidecar: %s", e)
        return ""

    def _prepare_irodori_reference_wav(reference_wav: Optional[str]) -> tuple[Optional[str], Optional[Path]]:
        if not reference_wav:
            return None, None
        source_path = Path(reference_wav)
        if source_path.suffix.lower() == ".wav":
            return str(source_path), None

        converted_path = _output_dir() / f"_irodori_ref_{uuid4().hex[:8]}.wav"
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            command = [
                ffmpeg_path,
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                str(converted_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                detail = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
                raise RuntimeError(f"Irodori-TTS用の参照音声WAV変換に失敗しました。\n{detail[-800:]}")
            return str(converted_path), converted_path

        try:
            import librosa
            import soundfile as sf

            wav_np, sr = librosa.load(str(source_path), sr=24000, mono=True)
            sf.write(str(converted_path), wav_np, sr, subtype="PCM_16")
            return str(converted_path), converted_path
        except Exception as e:
            raise RuntimeError(
                "Irodori-TTS用の参照音声WAV変換に失敗しました。"
                "m4a/mp3などを使う場合はffmpegをインストールするか、参照音声をWAVでアップロードしてください。"
                f" 詳細: {e}"
            ) from e

    def _generate_irodori_for_download(
        text: str,
        prefix: str,
        filename_hint: str = "",
        reference_wav: Optional[str] = None,
        style_features: Optional[list[str]] = None,
        control_instruction: str = "",
        voice_age: str = "",
        voice_gender: str = "",
        lora_adapter: str = "",
    ):
        temp_path = _output_dir() / f"_irodori_tmp_{uuid4().hex[:8]}.wav"
        styled_text = _build_irodori_style_text(text, style_features, control_instruction)
        caption = _build_irodori_caption(voice_age, voice_gender, style_features, control_instruction)
        prepared_reference, converted_reference = _prepare_irodori_reference_wav(reference_wav)
        try:
            sr, wav_np = demo.generate_irodori_audio(
                text_input=styled_text,
                output_wav_path=str(temp_path),
                reference_wav_path_input=prepared_reference,
                caption_input=caption,
                lora_adapter_path_input=lora_adapter,
            )
            output_path = _save_wav_for_download(sr, wav_np, prefix, filename_hint)
            return (sr, wav_np), output_path
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
                if converted_reference and converted_reference.exists():
                    converted_reference.unlink()
            except OSError:
                pass

    def _prepare_qwen3_reference_wav(reference_wav: Optional[str]) -> tuple[Optional[str], Optional[Path]]:
        if not reference_wav:
            return None, None
        source_path = Path(reference_wav)
        if source_path.suffix.lower() == ".wav":
            return str(source_path), None

        converted_path = _output_dir() / f"_qwen3_ref_{uuid4().hex[:8]}.wav"
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            command = [
                ffmpeg_path,
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                str(converted_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                detail = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
                raise RuntimeError(f"Qwen3-TTS用の参照音声WAV変換に失敗しました。\n{detail[-800:]}")
            return str(converted_path), converted_path

        try:
            import librosa
            import soundfile as sf

            wav_np, sr = librosa.load(str(source_path), sr=24000, mono=True)
            sf.write(str(converted_path), wav_np, sr, subtype="PCM_16")
            return str(converted_path), converted_path
        except Exception as e:
            raise RuntimeError(
                "Qwen3-TTS用の参照音声WAV変換に失敗しました。"
                "m4a/mp3などを使う場合はffmpegをインストールするか、参照音声をWAVでアップロードしてください。"
                f" 詳細: {e}"
            ) from e

    def _generate_qwen3_for_download(
        *,
        mode: str,
        text: str,
        prefix: str,
        filename_hint: str = "",
        target_language: str = "日本語",
        instruct: str = "",
        reference_wav: Optional[str] = None,
        reference_text: str = "",
    ):
        temp_path = _output_dir() / f"_qwen3_tmp_{uuid4().hex[:8]}.wav"
        prepared_reference, converted_reference = _prepare_qwen3_reference_wav(reference_wav)
        try:
            sr, wav_np = demo.generate_qwen3_audio(
                mode=mode,
                text_input=text,
                output_wav_path=str(temp_path),
                language_input=_qwen3_language_name(target_language),
                instruct_input=instruct,
                reference_wav_path_input=prepared_reference,
                reference_text_input=reference_text,
            )
            output_path = _save_wav_for_download(sr, wav_np, prefix, filename_hint)
            _write_reference_text_sidecar(output_path, text)
            return (sr, wav_np), output_path
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
                if converted_reference and converted_reference.exists():
                    converted_reference.unlink()
            except OSError:
                pass

    def _engine_status(engine_label: str):
        if _engine_is_irodori(engine_label):
            return demo.irodori_status()
        if _engine_is_qwen3(engine_label):
            return demo.qwen3_status()
        return "VoxCPM2を使用します。多言語、声のデザイン、声のクローン、高精度クローンに対応しています。"

    def _app_header_html(engine_label: str):
        if _engine_is_irodori(engine_label):
            logo_html = '<div class="text-logo">Irodori-TTS<span>日本語TTS</span></div>'
            engine_label_text = "Irodori-TTS"
        elif _engine_is_qwen3(engine_label):
            logo_html = '<div class="text-logo">VDC<span>Qwen3-TTS 簡易連携</span></div>'
            engine_label_text = "VoiceDesignCloner連携"
        else:
            logo_html = '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM2 Logo">'
            engine_label_text = "VoxCPM2"
        return (
            '<div class="logo-container">'
            f'<div class="app-brand">{logo_html}'
            '<div class="brand-copy">'
            "<h1>JP Voice Studio</h1>"
            "<p>日本語音声生成・声クローン統合ツール</p>"
            "</div></div>"
            f'<div class="engine-pill">使用中: {engine_label_text}</div>'
            "</div>"
        )

    def _engine_visibility_updates(engine_label: str):
        is_irodori = _engine_is_irodori(engine_label)
        is_qwen3 = _engine_is_qwen3(engine_label)
        voxcpm_only = gr.update(visible=not is_irodori and not is_qwen3)
        irodori_only = gr.update(visible=is_irodori)
        qwen3_only = gr.update(visible=is_qwen3)
        not_irodori = gr.update(visible=not is_irodori)
        supported_hifi = gr.update(visible=not is_irodori and not is_qwen3)
        unsupported_hifi = gr.update(visible=is_irodori or is_qwen3)
        return (
            _app_header_html(engine_label),
            _engine_status(engine_label),
            not_irodori,
            gr.update(visible=True),
            not_irodori,
            not_irodori,
            not_irodori,
            voxcpm_only,
            qwen3_only,
            gr.update(visible=not is_qwen3),
            qwen3_only,
            irodori_only,
            irodori_only,
            irodori_only,
            qwen3_only,
            qwen3_only,
            qwen3_only,
            not_irodori,
            voxcpm_only,
            voxcpm_only,
            voxcpm_only,
            voxcpm_only,
            supported_hifi,
            supported_hifi,
            unsupported_hifi,
        )

    _engine_tab_visibility_js = """
    (engineLabel) => {
        const apply = () => {
            const isIrodori = String(engineLabel || "").startsWith("Irodori-TTS");
            const label = String(engineLabel || "");
            const isQwen3 = label.startsWith("VoiceDesignCloner連携") || label.startsWith("Qwen3-TTS");
            const hideHifi = isIrodori || isQwen3;
            const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
            const hifiTab = tabs.find((tab) => (tab.textContent || "").includes("高精度クローン"));
            if (!hifiTab) return;
            hifiTab.style.display = hideHifi ? "none" : "";
            hifiTab.setAttribute("aria-hidden", hideHifi ? "true" : "false");
            if (hideHifi && hifiTab.getAttribute("aria-selected") === "true") {
                const fallbackTab =
                    tabs.find((tab) => (tab.textContent || "").includes("声のクローン")) ||
                    tabs.find((tab) => (tab.textContent || "").includes("声のデザイン"));
                if (fallbackTab) fallbackTab.click();
            }
        };
        apply();
        setTimeout(apply, 100);
        setTimeout(apply, 500);
        setTimeout(apply, 1200);
        return [engineLabel];
    }
    """

    def _generate_design(
        engine_label: str,
        text: str,
        voice_age: str,
        voice_gender: str,
        voice_features: Optional[list[str]],
        control_instruction: str,
        intonation_instruction: str,
        word_accent_instruction: str,
        target_language: str,
        irodori_lora_adapter: str,
        filename_hint: str,
        cfg_value: float,
        do_normalize: bool,
        dit_steps: int,
    ):
        if _engine_is_irodori(engine_label):
            _ensure_irodori_japanese(target_language)
            audio, output_path = _generate_irodori_for_download(
                text=text,
                prefix="irodori_design",
                filename_hint=filename_hint,
                style_features=voice_features,
                control_instruction="",
                voice_age=voice_age,
                voice_gender=voice_gender,
                lora_adapter=irodori_lora_adapter,
            )
            return audio, output_path, gr.update(choices=_list_voice_design_history(), value=None)

        if _engine_is_qwen3(engine_label):
            control_prompt = _combine_voice_profile_prompt(
                voice_age,
                voice_gender,
                voice_features,
                control_instruction,
            )
            instruct = " ".join(
                part
                for part in (
                    control_prompt,
                    _build_intonation_prompt(intonation_instruction),
                    _build_word_accent_prompt(word_accent_instruction),
                )
                if part
            )
            audio, output_path = _generate_qwen3_for_download(
                mode="design",
                text=text,
                prefix="qwen3_design",
                filename_hint=filename_hint,
                target_language=target_language,
                instruct=instruct,
            )
            return audio, output_path, gr.update(choices=_list_voice_design_history(), value=output_path)

        control_prompt = _combine_voice_profile_prompt(
            voice_age,
            voice_gender,
            voice_features,
            control_instruction,
        )
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction=control_prompt,
            intonation_instruction=intonation_instruction,
            word_accent_instruction=word_accent_instruction,
            target_language=target_language,
            reference_wav_path_input=None,
            prompt_text="",
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=False,
            inference_timesteps=int(dit_steps),
        )
        output_path = _save_wav_for_download(sr, wav_np, "voice_design", filename_hint)
        return (sr, wav_np), output_path, gr.update(choices=_list_voice_design_history(), value=output_path)

    def _generate_voice_gacha(
        engine_label: str,
        text: str,
        voice_age: str,
        voice_gender: str,
        voice_features: Optional[list[str]],
        control_instruction: str,
        intonation_instruction: str,
        word_accent_instruction: str,
        target_language: str,
        filename_hint: str,
        candidate_count: int,
    ):
        if not _engine_is_qwen3(engine_label):
            raise ValueError("声ガチャはVoiceDesignCloner連携（Qwen3-TTS・簡易）で利用できます。")

        count = max(1, min(int(candidate_count or 2), 4))
        control_prompt = _combine_voice_profile_prompt(
            voice_age,
            voice_gender,
            voice_features,
            control_instruction,
        )
        instruct = " ".join(
            part
            for part in (
                control_prompt,
                _build_intonation_prompt(intonation_instruction),
                _build_word_accent_prompt(word_accent_instruction),
            )
            if part
        )
        base_name = _sanitize_filename(filename_hint) or "gacha"
        audio_updates = []
        file_updates = []
        generated_paths: list[str] = []
        for index in range(4):
            if index < count:
                audio, output_path = _generate_qwen3_for_download(
                    mode="design",
                    text=text,
                    prefix="qwen3_gacha",
                    filename_hint=f"{base_name}_{index + 1:02d}",
                    target_language=target_language,
                    instruct=instruct,
                )
                audio_updates.append(gr.update(value=audio, visible=True))
                file_updates.append(gr.update(value=output_path, visible=True))
                generated_paths.append(output_path)
            else:
                audio_updates.append(gr.update(value=None, visible=False))
                file_updates.append(gr.update(value=None, visible=False))

        selected_value = generated_paths[0] if generated_paths else None
        status = (
            f"{len(generated_paths)}件の候補を生成しました。気に入った候補はWAVを確認し、"
            "履歴から別セリフ生成や声のクローンに使えます。"
        )
        return (
            *audio_updates,
            *file_updates,
            gr.update(choices=_list_voice_design_history(), value=selected_value),
            status,
        )

    def _generate_qwen3_design_candidates(
        engine_label: str,
        text: str,
        voice_age: str,
        voice_gender: str,
        voice_features: Optional[list[str]],
        control_instruction: str,
        intonation_instruction: str,
        word_accent_instruction: str,
        target_language: str,
        filename_hint: str,
        candidate_count: int,
    ):
        if not _engine_is_qwen3(engine_label):
            raise ValueError("Qwen3-TTSの生成数指定はVoiceDesignCloner連携（Qwen3-TTS・簡易）で利用できます。")

        count = max(1, min(int(candidate_count or 1), 4))
        control_prompt = _combine_voice_profile_prompt(
            voice_age,
            voice_gender,
            voice_features,
            control_instruction,
        )
        instruct = " ".join(
            part
            for part in (
                control_prompt,
                _build_intonation_prompt(intonation_instruction),
                _build_word_accent_prompt(word_accent_instruction),
            )
            if part
        )
        base_name = _sanitize_filename(filename_hint) or ("single" if count == 1 else "candidates")
        audio_updates = []
        file_updates = []
        generated: list[tuple[Tuple[int, np.ndarray], str]] = []

        for index in range(count):
            output_hint = base_name if count == 1 else f"{base_name}_{index + 1:02d}"
            audio, output_path = _generate_qwen3_for_download(
                mode="design",
                text=text,
                prefix="qwen3_design" if count == 1 else "qwen3_candidate",
                filename_hint=output_hint,
                target_language=target_language,
                instruct=instruct,
            )
            generated.append((audio, output_path))

        for index in range(4):
            if count > 1 and index < len(generated):
                audio, output_path = generated[index]
                audio_updates.append(gr.update(value=audio, visible=True))
                file_updates.append(gr.update(value=output_path, visible=True))
            else:
                audio_updates.append(gr.update(value=None, visible=False))
                file_updates.append(gr.update(value=None, visible=False))

        main_audio, main_path = generated[0]
        if count == 1:
            status = "1件生成しました。生成されたWAVは履歴から再利用できます。"
        else:
            status = (
                f"{count}件の候補を生成しました。上の「生成された音声」には候補1を表示しています。"
                "気に入った候補はWAVを確認し、履歴から別セリフ生成や声のクローンに使えます。"
            )
        return (
            main_audio,
            main_path,
            *audio_updates,
            *file_updates,
            gr.update(choices=_list_voice_design_history(), value=main_path),
            status,
        )

    def _refresh_voice_design_history():
        return _history_dropdown_update()

    def _delete_voice_design_history(history_wav: Optional[str]):
        if not history_wav:
            message = "削除する履歴ファイルを選んでください。"
            return gr.update(), gr.update(), gr.update(), message
        output_dir = _output_dir().resolve()
        target = Path(history_wav).resolve()
        if output_dir not in target.parents:
            raise ValueError("現在の保存先フォルダ内の履歴ファイルだけ削除できます。")
        if target.exists():
            target.unlink()
            message = f"削除しました: {target.name}"
        else:
            message = "ファイルは既に存在しません。履歴を更新しました。"
        history_update = _history_dropdown_update()
        return history_update, history_update, history_update, message

    def _delete_voice_design_history_single(history_wav: Optional[str]):
        history_update, _, _, message = _delete_voice_design_history(history_wav)
        return history_update, message

    def _resolve_reference_audio(
        uploaded_ref: Optional[str],
        history_ref: Optional[str],
        feature_label: str,
    ) -> str:
        if uploaded_ref:
            return uploaded_ref
        if history_ref:
            return history_ref
        raise ValueError(f"{feature_label}には参照音声をアップロードするか、声のデザイン履歴から選んでください。")

    def _generate_from_design_history(
        engine_label: str,
        history_wav: Optional[str],
        text: str,
        target_language: str,
        irodori_lora_adapter: str,
        filename_hint: str,
        cfg_value: float,
        do_normalize: bool,
        dit_steps: int,
    ):
        if not history_wav:
            raise ValueError("再利用する声を履歴から選んでください。")
        if _engine_is_irodori(engine_label):
            _ensure_irodori_japanese(target_language)
            return _generate_irodori_for_download(
                text=text,
                prefix="irodori_history_reuse",
                filename_hint=filename_hint,
                reference_wav=history_wav,
                lora_adapter=irodori_lora_adapter,
            )
        if _engine_is_qwen3(engine_label):
            reference_text = _read_reference_text_sidecar(history_wav)
            if not reference_text:
                raise ValueError(
                    "Qwen3-TTSで履歴の声を再利用するには、その履歴WAVの横に参照テキスト（.txt）が必要です。"
                    "Qwen3-TTSで新しく声のデザインを生成してから再利用してください。"
                )
            return _generate_qwen3_for_download(
                mode="clone",
                text=text,
                prefix="qwen3_history_reuse",
                filename_hint=filename_hint,
                target_language=target_language,
                reference_wav=history_wav,
                reference_text=reference_text,
            )
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction="",
            intonation_instruction="",
            word_accent_instruction="",
            target_language=target_language,
            reference_wav_path_input=history_wav,
            prompt_text="",
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=False,
            inference_timesteps=int(dit_steps),
        )
        return (sr, wav_np), _save_wav_for_download(sr, wav_np, "voice_design_reuse", filename_hint)

    def _generate_clone(
        engine_label: str,
        text: str,
        voice_age: str,
        voice_gender: str,
        voice_features: Optional[list[str]],
        control_instruction: str,
        intonation_instruction: str,
        word_accent_instruction: str,
        ref_wav: Optional[str],
        history_wav: Optional[str],
        qwen3_ref_text: str,
        target_language: str,
        irodori_lora_adapter: str,
        filename_hint: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
    ):
        ref_source = _resolve_reference_audio(ref_wav, history_wav, "声のクローン")
        if _engine_is_irodori(engine_label):
            _ensure_irodori_japanese(target_language)
            return _generate_irodori_for_download(
                text=text,
                prefix="irodori_clone",
                filename_hint=filename_hint,
                reference_wav=ref_source,
                style_features=voice_features,
                control_instruction="",
                voice_age=voice_age,
                voice_gender=voice_gender,
                lora_adapter=irodori_lora_adapter,
            )
        if _engine_is_qwen3(engine_label):
            reference_text = (qwen3_ref_text or "").strip() or _read_reference_text_sidecar(history_wav)
            if not reference_text:
                raise ValueError(
                    "Qwen3-TTSの声のクローンには、参照音声の文字起こしが必要です。"
                    "参照音声で実際に話している内容を入力してください。"
                )
            return _generate_qwen3_for_download(
                mode="clone",
                text=text,
                prefix="qwen3_clone",
                filename_hint=filename_hint,
                target_language=target_language,
                reference_wav=ref_source,
                reference_text=reference_text,
            )
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction=control_instruction,
            intonation_instruction=intonation_instruction,
            word_accent_instruction=word_accent_instruction,
            target_language=target_language,
            reference_wav_path_input=ref_source,
            prompt_text="",
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
        )
        return (sr, wav_np), _save_wav_for_download(sr, wav_np, "voice_clone", filename_hint)

    def _generate_qwen3_corpus_batch(
        engine_label: str,
        ref_wav: Optional[str],
        history_wav: Optional[str],
        qwen3_ref_text: str,
        target_language: str,
        corpus_text: str,
        corpus_file: Optional[str],
        max_lines: int,
        folder_name: str,
        target_sr: int,
    ):
        if not _engine_is_qwen3(engine_label):
            raise ValueError("コーパス一括音声化はVoiceDesignCloner連携（Qwen3-TTS・簡易）で利用できます。")

        ref_source = _resolve_reference_audio(ref_wav, history_wav, "コーパス一括音声化")
        reference_text = (qwen3_ref_text or "").strip() or _read_reference_text_sidecar(history_wav)
        if not reference_text:
            raise ValueError(
                "Qwen3-TTSのコーパス一括音声化には、参照音声の文字起こしが必要です。"
                "参照音声で実際に話している内容を入力してください。"
            )

        lines = _collect_corpus_lines(corpus_text, corpus_file, max_lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_base = _sanitize_filename(folder_name) or "corpus"
        output_dir = _output_dir() / f"qwen3_corpus_{folder_base}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        texts_path = output_dir / "_input_texts.txt"
        texts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        prepared_reference, converted_reference = _prepare_qwen3_reference_wav(ref_source)
        try:
            output_folder, raw_folder, text_list = demo.generate_qwen3_corpus(
                texts_file_path=str(texts_path),
                output_dir_path=str(output_dir),
                language_input=_qwen3_language_name(target_language),
                reference_wav_path_input=str(prepared_reference),
                reference_text_input=reference_text,
                target_sr=int(target_sr or 0),
            )
            status = (
                f"{len(lines)}文のコーパスを生成しました。\n\n"
                f"- WAV: `{raw_folder}`\n"
                f"- テキストリスト: `{text_list}`\n\n"
                "必要に応じて下の前処理でリサンプルや esd.list 生成を実行できます。"
            )
            return status, output_folder, text_list
        finally:
            try:
                if converted_reference and converted_reference.exists():
                    converted_reference.unlink()
            except OSError:
                pass

    def _generate_high_fidelity_clone(
        engine_label: str,
        text: str,
        ref_wav: Optional[str],
        history_wav: Optional[str],
        prompt_text_value: str,
        prevent_leading_mix: bool,
        intonation_instruction: str,
        word_accent_instruction: str,
        target_language: str,
        filename_hint: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
    ):
        ref_source = _resolve_reference_audio(ref_wav, history_wav, "高精度クローン")
        if _engine_is_irodori(engine_label):
            raise ValueError("Irodori-TTSは高精度クローン（参照音声+文字起こしの連続生成）には未対応です。声のクローンタブでIrodori-TTSを使ってください。")
        if _engine_is_qwen3(engine_label):
            raise ValueError("Qwen3-TTSはこの高精度クローンタブには未対応です。声のクローンタブで参照音声と文字起こしを指定してください。")
        if not prevent_leading_mix and not (prompt_text_value or "").strip():
            raise ValueError("文字起こしを使う場合は、参照音声の文字起こしが必要です。")
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction="",
            intonation_instruction=intonation_instruction,
            word_accent_instruction=word_accent_instruction,
            target_language=target_language,
            reference_wav_path_input=ref_source,
            prompt_text="" if prevent_leading_mix else prompt_text_value,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
        )
        prefix = "high_fidelity_safe_clone" if prevent_leading_mix else "high_fidelity_clone"
        return (sr, wav_np), _save_wav_for_download(sr, wav_np, prefix, filename_hint)

    def _transcribe_reference(audio_path: Optional[str], history_wav: Optional[str] = None):
        try:
            ref_source = _resolve_reference_audio(audio_path, history_wav, "自動文字起こし")
        except ValueError:
            return gr.update(), "参照音声をアップロードするか、声のデザイン履歴から選んでから、自動文字起こしを試してください。"
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(ref_source)
            logger.info(f"ASR result: {asr_text[:60]}...")
            asr_text = (asr_text or "").strip()
            if not asr_text:
                return gr.update(value=""), "文字起こし結果が空でした。参照音声の内容を手入力してください。"
            return gr.update(value=asr_text), "自動文字起こしが完了しました。内容を確認して、必要なら修正してください。"
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return (
                gr.update(),
                f"自動文字起こしに失敗しました。参照音声の内容を手入力または貼り付けしてください。詳細: {e}",
            )

    def _language_dropdown():
        return gr.Dropdown(
            choices=_LANGUAGE_LABELS,
            value="日本語",
            label="発話言語",
            info="翻訳は行いません。選んだ言語で読み上げテキストを入力してください。",
        )

    def _word_accent_textbox(value: str = ""):
        return gr.Textbox(
            value=value,
            label="現在のアクセント指定",
            placeholder="例:\n苺=平坦\n雨=頭高\n橋=尾高",
            lines=4,
            info="1行に1語ずつ「単語=平坦」「単語=語尾上げ」「単語=頭高」のように指定します。",
        )

    def _set_word_accent(accent_text: str, target: str, accent: str) -> str:
        accent_text = accent_text or ""
        target = (target or "").strip()
        if not target:
            return accent_text
        target = target.strip(" 「」『』\"'")
        next_line = f"{target}={accent}"
        kept_lines = []
        for line in accent_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            word = re.split(r"[=：:]", stripped, maxsplit=1)[0].strip(" 「」『』\"'")
            if word != target:
                kept_lines.append(stripped)
        kept_lines.append(next_line)
        return "\n".join(kept_lines)

    def _replace_first(text: str, target: str, replacement: str) -> str:
        text = text or ""
        target = (target or "").strip()
        if not target or target not in text:
            return text
        return text.replace(target, replacement, 1)

    def _emphasize_word(text: str, target: str) -> str:
        target = (target or "").strip()
        return _replace_first(text, target, f"「{target}」")

    def _add_short_pause(text: str, target: str) -> str:
        target = (target or "").strip()
        return _replace_first(text, target, f"{target}、")

    def _add_long_pause(text: str, target: str) -> str:
        target = (target or "").strip()
        return _replace_first(text, target, f"{target}……")

    def _set_sentence_end(text: str, mark: str) -> str:
        text = (text or "").rstrip()
        if not text:
            return text
        return re.sub(r"[。.!！?？]*$", mark, text)

    def _add_prosody_controls(text_component: gr.Textbox):
        with gr.Accordion("記号で読み方を調整", open=False):
            gr.Markdown(
                "強調したい語を入力して、記号を追加できます。"
                "精密な音程指定ではありませんが、間や抑揚のヒントになります。"
            )
            target_word = gr.Textbox(
                label="調整したい語",
                placeholder="例: 重要",
                lines=1,
            )
            with gr.Row():
                emphasize_btn = gr.Button("「」強調", size="sm")
                short_pause_btn = gr.Button("、短い間", size="sm")
                long_pause_btn = gr.Button("……長い間", size="sm")
                strong_end_btn = gr.Button("！強く", size="sm")
                question_end_btn = gr.Button("？疑問", size="sm")

            emphasize_btn.click(
                fn=_emphasize_word,
                inputs=[text_component, target_word],
                outputs=[text_component],
                show_progress=False,
            )
            short_pause_btn.click(
                fn=_add_short_pause,
                inputs=[text_component, target_word],
                outputs=[text_component],
                show_progress=False,
            )
            long_pause_btn.click(
                fn=_add_long_pause,
                inputs=[text_component, target_word],
                outputs=[text_component],
                show_progress=False,
            )
            strong_end_btn.click(
                fn=lambda text: _set_sentence_end(text, "！"),
                inputs=[text_component],
                outputs=[text_component],
                show_progress=False,
            )
            question_end_btn.click(
                fn=lambda text: _set_sentence_end(text, "？"),
                inputs=[text_component],
                outputs=[text_component],
                show_progress=False,
            )

    def _add_word_accent_controls():
        with gr.Accordion("単語アクセントを指定", open=False):
            gr.Markdown(
                "アクセントを変えたい語を入力して、型を選びます。"
                "例: `イチゴ` + `語尾上げ` -> `イチゴ=語尾上げ`"
            )
            target_word = gr.Textbox(
                label="アクセントを指定したい語",
                placeholder="例: イチゴ",
                lines=1,
            )
            accent_text = _word_accent_textbox()
            with gr.Row():
                flat_btn = gr.Button("平坦", size="sm")
                rising_btn = gr.Button("語尾上げ", size="sm")
                atamadaka_btn = gr.Button("頭高", size="sm")
                nakadaka_btn = gr.Button("中高", size="sm")
                odaka_btn = gr.Button("尾高", size="sm")

            flat_btn.click(
                fn=lambda current, target: _set_word_accent(current, target, "平坦"),
                inputs=[accent_text, target_word],
                outputs=[accent_text],
                show_progress=False,
            )
            rising_btn.click(
                fn=lambda current, target: _set_word_accent(current, target, "語尾上げ"),
                inputs=[accent_text, target_word],
                outputs=[accent_text],
                show_progress=False,
            )
            atamadaka_btn.click(
                fn=lambda current, target: _set_word_accent(current, target, "頭高"),
                inputs=[accent_text, target_word],
                outputs=[accent_text],
                show_progress=False,
            )
            nakadaka_btn.click(
                fn=lambda current, target: _set_word_accent(current, target, "中高"),
                inputs=[accent_text, target_word],
                outputs=[accent_text],
                show_progress=False,
            )
            odaka_btn.click(
                fn=lambda current, target: _set_word_accent(current, target, "尾高"),
                inputs=[accent_text, target_word],
                outputs=[accent_text],
                show_progress=False,
            )
        return accent_text

    def _advanced_settings(include_denoise: bool = False, cfg_default: float = 2.0):
        with gr.Accordion("詳細設定", open=False):
            denoise_control = None
            if include_denoise:
                denoise_control = gr.Checkbox(
                    value=False,
                    label="参照音声のノイズ除去",
                    elem_classes=["switch-toggle"],
                    info="クローン前に参照音声を補正します。必要な場合だけ有効にしてください。",
                )
            normalize_control = gr.Checkbox(
                value=False,
                label="テキスト正規化",
                elem_classes=["switch-toggle"],
                info="数字、日付、省略表記などを読み上げ向けに整えます。",
            )
            cfg_control = gr.Slider(
                minimum=1.0,
                maximum=3.0,
                value=cfg_default,
                step=0.1,
                label="CFG（指示への追従度）",
                info="大きいほど指示や参照音声に寄せ、小さいほど生成の自由度が上がります。",
            )
            steps_control = gr.Slider(
                minimum=1,
                maximum=50,
                value=10,
                step=1,
                label="生成ステップ数",
                info="増やすと品質が上がる場合がありますが、生成は遅くなります。",
            )
        return denoise_control, normalize_control, cfg_control, steps_control

    _RECORDING_SCRIPT_PRESETS = {
        "落ち着いたナレーション": (
            "こんにちは。今日は音声生成のための参照音声を録音しています。"
            "普段の声で、少しゆっくり、はっきりと話します。"
            "静かな場所で録音すると、声の特徴がより伝わりやすくなります。"
        ),
        "自然な会話": (
            "こんにちは、調子はいかがですか。"
            "今日は新しい音声ツールを試しています。"
            "短い文章でも、自然な間を入れて話すと、声の雰囲気が分かりやすくなります。"
        ),
        "明るい案内": (
            "お知らせします。こちらは音声クローン用のテスト録音です。"
            "明るく聞き取りやすい声で、最後まで同じ距離を保って話します。"
            "ご利用ありがとうございます。"
        ),
        "聞き取り確認": (
            "数字の一、二、三、曜日の月曜日、火曜日、水曜日を読み上げます。"
            "短い言葉と長い言葉を混ぜて、声の高さや話す速さを確認します。"
            "これで録音を終了します。"
        ),
    }
    _RECORDING_SCRIPT_LABELS = list(_RECORDING_SCRIPT_PRESETS.keys())
    _DEFAULT_RECORDING_SCRIPT_LABEL = "落ち着いたナレーション"

    def _recording_script_for_preset(preset_label: str) -> str:
        return _RECORDING_SCRIPT_PRESETS.get(preset_label, _RECORDING_SCRIPT_PRESETS[_DEFAULT_RECORDING_SCRIPT_LABEL])

    def _copy_recording_script_to_prompt(script: str):
        script = (script or "").strip()
        if not script:
            return gr.update(), "録音原稿が空です。原稿を選ぶか入力してから反映してください。"
        return script, "録音原稿を参照音声の文字起こし欄へ入れました。録音で読んだ内容と一致しているか確認してください。"

    def _add_reference_recording_guide(open_default: bool = False):
        with gr.Accordion("参照音声を録音する", open=open_default):
            gr.Markdown(
                "目安は5〜30秒です。静かな場所で、BGMなし、1人の声だけを録音してください。"
                "声を作り込みすぎず、普段の話し方で読むと安定しやすくなります。"
            )
            preset = gr.Dropdown(
                choices=_RECORDING_SCRIPT_LABELS,
                value=_DEFAULT_RECORDING_SCRIPT_LABEL,
                label="録音原稿プリセット",
            )
            script = gr.Textbox(
                value=_recording_script_for_preset(_DEFAULT_RECORDING_SCRIPT_LABEL),
                label="録音で読む原稿",
                lines=5,
            )
            preset.change(
                fn=_recording_script_for_preset,
                inputs=[preset],
                outputs=[script],
                show_progress=False,
                api_name=None,
                api_visibility="private",
            )
        return preset, script

    with gr.Blocks(title="JP Voice Studio") as interface:
        app_header = gr.HTML(_app_header_html(_ENGINE_VOXCPM))
        gr.Markdown("**用途に合わせてモードを選んでください。** 各画面には、その生成方法に必要な入力だけを表示しています。")

        with gr.Accordion("音声エンジン", open=True):
            engine_selector = gr.Radio(
                choices=_ENGINE_LABELS,
                value=_ENGINE_VOXCPM,
                label="音声エンジン",
                info="Irodori-TTSは日本語専用の実験対応です。未セットアップ時は案内を表示します。",
            )
            engine_status = gr.Markdown(_engine_status(_ENGINE_VOXCPM))

        with gr.Accordion("保存先フォルダ", open=True):
            output_dir_global = gr.Textbox(
                value=str(_output_dir()),
                label="保存先フォルダ",
                info="生成したWAVと声のデザイン履歴を保存するフォルダです。相対パスも指定できます。",
                lines=1,
            )
            with gr.Row():
                output_dir_apply = gr.Button("保存先を変更", variant="secondary")
                output_dir_open = gr.Button("フォルダを開く", variant="secondary")
            output_dir_status = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("声のデザイン") as design_tab:
                gr.Markdown(
                    "参照音声を使わず、声の雰囲気を文章で指定して新しい声を作ります。"
                    "男性声・女性声・話す速さなどの日本語指定は、内部でモデル向けの声質タグに補強されます。"
                )
                with gr.Row():
                    with gr.Column():
                        design_language = _language_dropdown()
                        with gr.Accordion("声の基本設定", open=True):
                            with gr.Row() as design_voice_age_gender_row:
                                design_voice_age = gr.Dropdown(
                                    choices=_VOICE_AGE_LABELS,
                                    value="大人",
                                    label="年齢",
                                    info="自由入力より安定しやすい声質タグとして反映します。",
                                )
                                design_voice_gender = gr.Dropdown(
                                    choices=_VOICE_GENDER_LABELS,
                                    value="男性",
                                    label="性別",
                                    info="声質の方向性として反映します。",
                                )
                            design_voice_features = gr.CheckboxGroup(
                                choices=_VOICE_FEATURE_LABELS,
                                value=["落ち着いた", "ナレーション", "聞き取りやすい", "ゆっくり"],
                                label="特徴",
                                info="複数選べます。矛盾する特徴を同時に選ぶと効果が弱くなることがあります。",
                            )
                        design_control = gr.Textbox(
                            value="低めの落ち着いた日本語の男性ナレーション。大人の男性声で、聞き取りやすく、少しゆっくり話す。",
                            label="声の指示",
                            placeholder="例: 低めの男性ナレーション / やさしい女性の声 / 元気なキャラクター声",
                            lines=3,
                        )
                        with gr.Group(visible=False) as design_irodori_lora_group:
                            design_irodori_lora = gr.Dropdown(
                                choices=_list_irodori_lora_adapters(),
                                value="",
                                label="Irodori LoRAアダプタ",
                                info="学習済みLoRAをIrodori-TTS推論に適用します。",
                            )
                            design_irodori_lora_refresh = gr.Button("LoRA一覧を更新", variant="secondary", size="sm")
                        design_intonation = gr.State("")
                        with gr.Group() as design_word_accent_group:
                            design_word_accent = _add_word_accent_controls()
                        design_text = gr.Textbox(
                            value=DEFAULT_TARGET_TEXT,
                            label="読み上げテキスト",
                            lines=5,
                        )
                        design_filename = gr.Textbox(
                            value="",
                            label="保存ファイル名（任意）",
                            placeholder="例: calm_male_narration",
                            lines=1,
                        )
                        with gr.Group() as design_prosody_group:
                            _add_prosody_controls(design_text)
                        with gr.Group() as design_advanced_group:
                            _, design_normalize, design_cfg, design_steps = _advanced_settings(
                                include_denoise=False,
                                cfg_default=2.6,
                            )
                        design_btn = gr.Button("この声を生成", variant="primary", size="lg")
                        with gr.Group(visible=False) as design_qwen3_single_group:
                            gr.Markdown(
                                "**Qwen3-TTS 生成**\n\n"
                                "左カラムの声の指示・読み上げテキスト・発話言語を使って生成します。"
                                "生成数を1にすると単発生成、2以上にすると複数候補の比較になります。"
                            )
                            design_qwen3_count = gr.Dropdown(
                                choices=[1, 2, 3, 4],
                                value=1,
                                label="生成数",
                                info="候補を増やすほど生成時間とVRAM使用時間が増えます。",
                            )
                            design_qwen3_generate_btn = gr.Button("指定数を生成", variant="primary", size="lg")
                    with gr.Column():
                        design_output = gr.Audio(label="生成された音声")
                        design_file = gr.File(label="WAVダウンロード", interactive=False)
                        with gr.Group(visible=False) as design_qwen3_gacha_group:
                            with gr.Accordion("生成候補", open=True):
                                gr.Markdown(
                                    "生成数を2以上にした場合、候補がここに並びます。"
                                    "生成数が1の場合は上の「生成された音声」だけを使います。"
                                )
                                design_gacha_status = gr.Markdown("")
                                design_gacha_audio_1 = gr.Audio(label="候補 1", visible=False)
                                design_gacha_file_1 = gr.File(label="候補 1 WAV", interactive=False, visible=False)
                                design_gacha_audio_2 = gr.Audio(label="候補 2", visible=False)
                                design_gacha_file_2 = gr.File(label="候補 2 WAV", interactive=False, visible=False)
                                design_gacha_audio_3 = gr.Audio(label="候補 3", visible=False)
                                design_gacha_file_3 = gr.File(label="候補 3 WAV", interactive=False, visible=False)
                                design_gacha_audio_4 = gr.Audio(label="候補 4", visible=False)
                                design_gacha_file_4 = gr.File(label="候補 4 WAV", interactive=False, visible=False)
                        gr.Markdown(
                            "**使い方**\n\n"
                            "1. 声の指示に、声質や話し方を書きます。\n"
                            "2. 読み上げたい文章を入力します。\n"
                            "3. 表示されている生成ボタンを押します。"
                        )
                        with gr.Accordion("声のデザイン履歴から再利用", open=True):
                            gr.Markdown(
                                "声のデザインで生成した音声を参照音声として使い、"
                                "同じ声質に近い声で別のセリフを生成します。"
                            )
                            design_output_dir = gr.Textbox(
                                value=str(_output_dir()),
                                label="保存先フォルダ",
                                interactive=False,
                            )
                            design_history = gr.Dropdown(
                                choices=_list_voice_design_history(),
                                value=(_list_voice_design_history()[0][1] if _list_voice_design_history() else None),
                                label="再利用する声",
                                info="新しく声を生成すると、この履歴に追加されます。",
                            )
                            design_history_refresh = gr.Button("履歴を更新", variant="secondary", size="sm")
                            design_history_delete = gr.Button("選択した履歴を削除", variant="stop", size="sm")
                            design_history_status = gr.Markdown("")
                            design_reuse_text = gr.Textbox(
                                value="この声で、別のセリフも読んでみます。",
                                label="この声で読み上げるテキスト",
                                lines=4,
                            )
                            with gr.Group(visible=False) as design_reuse_irodori_lora_group:
                                design_reuse_irodori_lora = gr.Dropdown(
                                    choices=_list_irodori_lora_adapters(),
                                    value="",
                                    label="Irodori LoRAアダプタ",
                                )
                                design_reuse_irodori_lora_refresh = gr.Button("LoRA一覧を更新", variant="secondary", size="sm")
                            design_reuse_filename = gr.Textbox(
                                value="",
                                label="保存ファイル名（任意）",
                                placeholder="例: calm_male_line2",
                                lines=1,
                            )
                            design_reuse_btn = gr.Button("履歴の声で生成", variant="primary")
                            design_reuse_output = gr.Audio(label="履歴の声で生成された音声")
                            design_reuse_file = gr.File(label="WAVダウンロード", interactive=False)

                design_btn.click(
                    fn=_generate_design,
                    inputs=[
                        engine_selector,
                        design_text,
                        design_voice_age,
                        design_voice_gender,
                        design_voice_features,
                        design_control,
                        design_intonation,
                        design_word_accent,
                        design_language,
                        design_irodori_lora,
                        design_filename,
                        design_cfg,
                        design_normalize,
                        design_steps,
                    ],
                    outputs=[design_output, design_file, design_history],
                    show_progress=True,
                    api_name="design",
                )
                design_qwen3_generate_btn.click(
                    fn=_generate_qwen3_design_candidates,
                    inputs=[
                        engine_selector,
                        design_text,
                        design_voice_age,
                        design_voice_gender,
                        design_voice_features,
                        design_control,
                        design_intonation,
                        design_word_accent,
                        design_language,
                        design_filename,
                        design_qwen3_count,
                    ],
                    outputs=[
                        design_output,
                        design_file,
                        design_gacha_audio_1,
                        design_gacha_audio_2,
                        design_gacha_audio_3,
                        design_gacha_audio_4,
                        design_gacha_file_1,
                        design_gacha_file_2,
                        design_gacha_file_3,
                        design_gacha_file_4,
                        design_history,
                        design_gacha_status,
                    ],
                    show_progress=True,
                    api_name="qwen3_design_candidates",
                )
                design_history_refresh.click(
                    fn=_refresh_voice_design_history,
                    inputs=[],
                    outputs=[design_history],
                    show_progress=False,
                )
                design_irodori_lora_refresh.click(
                    fn=_lora_adapter_dropdown_update,
                    inputs=[],
                    outputs=[design_irodori_lora],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                design_reuse_irodori_lora_refresh.click(
                    fn=_lora_adapter_dropdown_update,
                    inputs=[],
                    outputs=[design_reuse_irodori_lora],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                design_history_delete.click(
                    fn=_delete_voice_design_history_single,
                    inputs=[design_history],
                    outputs=[design_history, design_history_status],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                design_reuse_btn.click(
                    fn=_generate_from_design_history,
                    inputs=[
                        engine_selector,
                        design_history,
                        design_reuse_text,
                        design_language,
                        design_reuse_irodori_lora,
                        design_reuse_filename,
                        design_cfg,
                        design_normalize,
                        design_steps,
                    ],
                    outputs=[design_reuse_output, design_reuse_file],
                    show_progress=True,
                    api_name="design_reuse",
                )

            with gr.Tab("声のクローン") as clone_tab:
                gr.Markdown("参照音声の声質をもとに、別の文章を読み上げます。選択中のエンジンに必要な追加項目だけを表示します。")
                with gr.Row():
                    with gr.Column():
                        clone_ref = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="参照音声（アップロード / マイク録音）",
                        )
                        _, clone_recording_script = _add_reference_recording_guide(open_default=True)
                        clone_history = gr.Dropdown(
                            choices=_list_voice_design_history(),
                            value=None,
                            label="声のデザイン履歴から選択（任意）",
                            info="参照音声をアップロードしていない場合、この履歴の声を使います。",
                        )
                        clone_history_refresh = gr.Button("履歴を更新", variant="secondary", size="sm")
                        with gr.Group(visible=False) as clone_qwen3_ref_text_group:
                            clone_qwen3_ref_text = gr.Textbox(
                                value="",
                                label="参照音声の文字起こし（Qwen3-TTS用）",
                                placeholder="参照音声で実際に話している内容を入力してください。例: こんにちは。今日は音声生成のテストをしています。",
                                lines=3,
                                info="Qwen3-TTSの声のクローンでは、参照音声と同じ内容の文字起こしが必要です。Qwen3の履歴を選んだ場合は保存済みテキストを自動利用できます。",
                            )
                        with gr.Group(visible=False) as clone_irodori_profile_group:
                            gr.Markdown("**Irodori声質ヒント**\n\n参照音声が優先されますが、年齢・性別・特徴を声質説明として補助的に渡します。")
                            with gr.Row():
                                clone_voice_age = gr.Dropdown(
                                    choices=_VOICE_AGE_LABELS,
                                    value="指定なし",
                                    label="年齢",
                                )
                                clone_voice_gender = gr.Dropdown(
                                    choices=_VOICE_GENDER_LABELS,
                                    value="指定なし",
                                    label="性別",
                                )
                            clone_voice_features = gr.CheckboxGroup(
                                choices=_VOICE_FEATURE_LABELS,
                                value=[],
                                label="特徴",
                            )
                            clone_irodori_lora = gr.Dropdown(
                                choices=_list_irodori_lora_adapters(),
                                value="",
                                label="Irodori LoRAアダプタ",
                                info="学習済みLoRAをIrodori-TTS推論に適用します。",
                            )
                            clone_irodori_lora_refresh = gr.Button("LoRA一覧を更新", variant="secondary", size="sm")
                        clone_language = _language_dropdown()
                        clone_control = gr.Textbox(
                            value="自然で聞き取りやすく、落ち着いた話し方",
                            label="声の指示（任意）",
                            placeholder="例: 少し明るく / ゆっくり丁寧に / 感情を抑えて",
                            lines=2,
                        )
                        clone_intonation = gr.State("")
                        with gr.Group() as clone_word_accent_group:
                            clone_word_accent = _add_word_accent_controls()
                        clone_text = gr.Textbox(
                            value="これは参照音声を使った声のクローン生成テストです。",
                            label="読み上げテキスト",
                            lines=5,
                        )
                        clone_filename = gr.Textbox(
                            value="",
                            label="保存ファイル名（任意）",
                            placeholder="例: cloned_voice_sample",
                            lines=1,
                        )
                        with gr.Group() as clone_prosody_group:
                            _add_prosody_controls(clone_text)
                        with gr.Group() as clone_advanced_group:
                            clone_denoise, clone_normalize, clone_cfg, clone_steps = _advanced_settings(include_denoise=True)
                        clone_btn = gr.Button("この声で生成", variant="primary", size="lg")
                        with gr.Group(visible=False) as clone_qwen3_corpus_group:
                            gr.Markdown(
                                "**コーパス一括音声化（簡易）**\n\n"
                                "選択中の参照音声または履歴の声で、1行1文のテキストをまとめてWAV化します。"
                                "出力は `raw/*.wav` と `Neutral.txt` です。"
                            )
                            clone_corpus_text = gr.Textbox(
                                value="今日は新しい音声モデルのテストをしています。\nこの声で複数の文章を読み上げます。\n自然で聞き取りやすい音声を目指します。",
                                label="コーパス本文（1行1文）",
                                lines=8,
                                placeholder="1行につき1文を入力してください。",
                            )
                            clone_corpus_file = gr.File(
                                label="コーパスTXT（任意）",
                                file_types=[".txt"],
                                type="filepath",
                            )
                            with gr.Row():
                                clone_corpus_limit = gr.Dropdown(
                                    choices=[10, 50, 100, 300, 1000],
                                    value=10,
                                    label="生成する文数",
                                    info="まずは10文程度で声質を確認するのがおすすめです。",
                                )
                                clone_corpus_target_sr = gr.Dropdown(
                                    choices=[24000, 44100, 48000],
                                    value=44100,
                                    label="出力サンプルレート",
                                )
                            clone_corpus_folder_name = gr.Textbox(
                                value="",
                                label="出力フォルダ名（任意）",
                                placeholder="例: my_character_corpus",
                                lines=1,
                            )
                            clone_corpus_btn = gr.Button("コーパスを一括生成", variant="secondary")
                    with gr.Column():
                        clone_output = gr.Audio(label="生成された音声")
                        clone_file = gr.File(label="WAVダウンロード", interactive=False)
                        clone_output_dir = gr.Textbox(
                            value=str(_output_dir()),
                            label="保存先フォルダ",
                            interactive=False,
                        )
                        clone_history_delete = gr.Button("選択した履歴を削除", variant="stop", size="sm")
                        clone_history_status = gr.Markdown("")
                        with gr.Group(visible=False) as clone_qwen3_corpus_result_group:
                            clone_corpus_status = gr.Markdown("")
                            clone_corpus_output_dir = gr.Textbox(
                                value="",
                                label="生成したコーパス出力フォルダ",
                                interactive=False,
                            )
                            clone_corpus_text_list_file = gr.File(label="Neutral.txt", interactive=False)
                            clone_corpus_tools_dir = gr.Textbox(
                                value="",
                                label="前処理するコーパスフォルダ（貼り付け可）",
                                placeholder="例: D:\\AIProduct\\VoxCPM\\outputs\\qwen3_corpus_sample_20260603_120000",
                                interactive=True,
                            )
                            clone_corpus_open_dir = gr.Button("前処理フォルダを開く", variant="secondary", size="sm")
                            gr.Markdown(
                                "**Style-Bert-VITS2向け前処理**\n\n"
                                "`raw/*.wav` と `Neutral.txt` から、リサンプル済みWAVと `esd.list` を作成します。"
                            )
                            with gr.Row():
                                clone_corpus_resample_sr = gr.Dropdown(
                                    choices=[44100, 48000, 24000, 22050],
                                    value=44100,
                                    label="リサンプル先Hz",
                                )
                                clone_corpus_resample_btn = gr.Button("rawをresampledへ変換", variant="secondary")
                            clone_corpus_resample_status = gr.Textbox(
                                value="",
                                label="リサンプル結果",
                                interactive=False,
                                lines=3,
                            )
                            with gr.Row():
                                clone_corpus_speaker = gr.Textbox(
                                    value="",
                                    label="話者名",
                                    placeholder="空欄ならフォルダ名を使います",
                                    lines=1,
                                )
                                clone_corpus_esd_lang = gr.Dropdown(
                                    choices=["JP", "EN", "ZH"],
                                    value="JP",
                                    label="esd.list言語コード",
                                )
                            clone_corpus_esd_btn = gr.Button("esd.listを生成", variant="secondary")
                            clone_corpus_esd_status = gr.Textbox(
                                value="",
                                label="esd.list生成結果",
                                interactive=False,
                                lines=3,
                            )
                            clone_corpus_esd_file = gr.File(label="esd.list", interactive=False)
                            gr.Markdown(
                                "**Irodori LoRA学習の流れ**\n\n"
                                "1. コーパスを生成し、必要なら `resampled` へ変換します。\n"
                                "2. `LoRA学習データを準備` で lab フォルダを作成します。\n"
                                "3. まずドライランでコマンドとパスを確認します。\n"
                                "4. 問題なければドライランを外し、短いステップ数から実学習します。\n"
                                "5. 学習後は Irodori-TTS を選び、声のデザインまたは声のクローンで LoRA アダプタを選びます。\n\n"
                                "テストは10文前後でも動作確認できます。声質として使う場合は、静かな音声で50文以上、できれば数百文あると安定しやすくなります。"
                            )
                            gr.Markdown(
                                "**Irodori-TTS LoRA学習データ準備**\n\n"
                                "生成済みコーパスをIrodori-TTSのLoRA学習で使う `lab/{話者}/{感情}` 形式へ変換します。"
                            )
                            with gr.Row():
                                clone_lora_speaker = gr.Textbox(
                                    value="",
                                    label="LoRA話者名",
                                    placeholder="例: honoka",
                                    lines=1,
                                )
                                clone_lora_emotion = gr.Textbox(
                                    value="Neutral",
                                    label="感情ラベル",
                                    placeholder="例: Neutral",
                                    lines=1,
                                )
                            clone_lora_wav_folder = gr.Dropdown(
                                choices=["raw", "resampled"],
                                value="resampled",
                                label="学習に使うWAVフォルダ",
                                info="resampledを使う場合は、先にリサンプルを実行してください。",
                            )
                            clone_lora_prepare_btn = gr.Button("1. LoRA学習データを準備", variant="secondary")
                            clone_lora_prepare_status = gr.Textbox(
                                value="",
                                label="LoRA学習データ準備結果",
                                interactive=False,
                                lines=6,
                            )
                            clone_lora_lab_dir = gr.Textbox(
                                value="",
                                label="LoRA labフォルダ",
                                interactive=False,
                            )
                            clone_lora_lab_text_file = gr.File(label="labテキスト", interactive=False)
                            clone_lora_jsonl_file = gr.File(label="training JSONL", interactive=False)
                            gr.Markdown(
                                "**LoRA学習実行（実験）**\n\n"
                                "既定ではドライランです。ドライランでは学習せず、実行されるコマンドだけを確認します。実学習はGPUを使うため、まず少ないステップ数で試してください。"
                            )
                            clone_lora_train_lab_dir = gr.Textbox(
                                value="",
                                label="学習するlabフォルダ（貼り付け可）",
                                placeholder="例: D:\\AIProduct\\VoxCPM\\outputs\\lora_data\\lab\\honoka\\Neutral",
                                interactive=True,
                            )
                            clone_lora_quality_btn = gr.Button("学習データをチェック", variant="secondary")
                            clone_lora_quality_status = gr.Textbox(
                                value="",
                                label="学習データ品質チェック",
                                interactive=False,
                                lines=12,
                            )
                            with gr.Row():
                                clone_lora_train_steps = gr.Number(
                                    value=50,
                                    label="学習ステップ数",
                                    info="動作確認は1〜50、本格調整は音声量を増やしてから段階的に上げます。",
                                    precision=0,
                                )
                                clone_lora_train_batch = gr.Number(
                                    value=1,
                                    label="バッチサイズ",
                                    info="VRAM不足時は1のままにしてください。",
                                    precision=0,
                                )
                                clone_lora_train_workers = gr.Number(
                                    value=0,
                                    label="ワーカー数",
                                    info="Windowsでは0が安定しやすいです。",
                                    precision=0,
                                )
                            clone_lora_train_lr = gr.Number(
                                value=0.0001,
                                label="学習率",
                                info="迷ったら既定値のままで始めてください。",
                            )
                            clone_lora_train_dry_run = gr.Checkbox(
                                value=True,
                                label="ドライラン（まずはオン推奨）",
                                info="オンの間は実学習しません。ログでパスを確認してから外します。",
                            )
                            with gr.Row():
                                clone_lora_train_btn = gr.Button("2. LoRA学習を開始", variant="primary")
                                clone_lora_train_stop_btn = gr.Button("停止", variant="stop")
                            clone_lora_train_status = gr.Textbox(
                                value="",
                                label="LoRA学習ステータス",
                                interactive=False,
                                lines=3,
                            )
                            clone_lora_train_log = gr.Textbox(
                                value="",
                                label="LoRA学習ログ",
                                interactive=False,
                                lines=12,
                            )
                            with gr.Row():
                                clone_lora_adapter_refresh = gr.Button("3. LoRAアダプタ一覧を更新", variant="secondary")
                                clone_lora_adapter_open_dir = gr.Button("LoRA保存フォルダを開く", variant="secondary")
                            clone_lora_adapter_status = gr.Textbox(
                                value=_lora_adapter_summary(),
                                label="学習済みLoRAアダプタ一覧",
                                interactive=False,
                                lines=7,
                            )
                        gr.Markdown(
                            "**使い方**\n\n"
                            "1. クローンしたい声の音声をアップロードするか、声のデザイン履歴から選びます。\n"
                            "2. 表示されている追加項目を入力します。Qwen3-TTSでは参照音声の文字起こしが必要です。\n"
                            "3. 読み上げテキストを入力して生成します。"
                        )

                clone_btn.click(
                    fn=_generate_clone,
                    inputs=[
                        engine_selector,
                        clone_text,
                        clone_voice_age,
                        clone_voice_gender,
                        clone_voice_features,
                        clone_control,
                        clone_intonation,
                        clone_word_accent,
                        clone_ref,
                        clone_history,
                        clone_qwen3_ref_text,
                        clone_language,
                        clone_irodori_lora,
                        clone_filename,
                        clone_cfg,
                        clone_normalize,
                        clone_denoise,
                        clone_steps,
                    ],
                    outputs=[clone_output, clone_file],
                    show_progress=True,
                    api_name="clone",
                )
                clone_corpus_btn.click(
                    fn=_generate_qwen3_corpus_batch,
                    inputs=[
                        engine_selector,
                        clone_ref,
                        clone_history,
                        clone_qwen3_ref_text,
                        clone_language,
                        clone_corpus_text,
                        clone_corpus_file,
                        clone_corpus_limit,
                        clone_corpus_folder_name,
                        clone_corpus_target_sr,
                    ],
                    outputs=[clone_corpus_status, clone_corpus_output_dir, clone_corpus_text_list_file],
                    show_progress=True,
                    api_name="qwen3_corpus_batch",
                )
                clone_corpus_open_dir.click(
                    fn=_open_existing_folder,
                    inputs=[clone_corpus_tools_dir],
                    outputs=[clone_corpus_status],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                clone_corpus_resample_btn.click(
                    fn=_resample_corpus_raw,
                    inputs=[clone_corpus_tools_dir, clone_corpus_resample_sr],
                    outputs=[clone_corpus_resample_status],
                    show_progress=True,
                    api_name="qwen3_corpus_resample",
                )
                clone_corpus_esd_btn.click(
                    fn=_generate_corpus_esd_list,
                    inputs=[clone_corpus_tools_dir, clone_corpus_speaker, clone_corpus_esd_lang],
                    outputs=[clone_corpus_esd_status, clone_corpus_esd_file],
                    show_progress=True,
                    api_name="qwen3_corpus_esd_list",
                )
                clone_lora_prepare_btn.click(
                    fn=_prepare_irodori_lora_data,
                    inputs=[
                        clone_corpus_tools_dir,
                        clone_lora_speaker,
                        clone_lora_emotion,
                        clone_lora_wav_folder,
                    ],
                    outputs=[
                        clone_lora_prepare_status,
                        clone_lora_lab_dir,
                        clone_lora_train_lab_dir,
                        clone_lora_lab_text_file,
                        clone_lora_jsonl_file,
                    ],
                    show_progress=True,
                    api_name="qwen3_prepare_irodori_lora_data",
                )
                clone_lora_quality_btn.click(
                    fn=_check_irodori_lora_lab_data,
                    inputs=[clone_lora_train_lab_dir],
                    outputs=[clone_lora_quality_status],
                    show_progress=True,
                    api_name="qwen3_check_irodori_lora_data",
                )
                lora_train_event = clone_lora_train_btn.click(
                    fn=_run_irodori_lora_training,
                    inputs=[
                        clone_lora_train_lab_dir,
                        clone_lora_train_steps,
                        clone_lora_train_batch,
                        clone_lora_train_workers,
                        clone_lora_train_lr,
                        clone_lora_train_dry_run,
                    ],
                    outputs=[clone_lora_train_status, clone_lora_train_log],
                    show_progress=True,
                    api_name="qwen3_run_irodori_lora_training",
                )
                lora_train_event.then(
                    fn=_refresh_lora_adapter_management,
                    inputs=[],
                    outputs=[
                        design_irodori_lora,
                        design_reuse_irodori_lora,
                        clone_irodori_lora,
                        clone_lora_adapter_status,
                    ],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                clone_lora_train_stop_btn.click(
                    fn=None,
                    cancels=[lora_train_event],
                    api_name=None,
                    api_visibility="private",
                )
                clone_lora_adapter_refresh.click(
                    fn=_refresh_lora_adapter_management,
                    inputs=[],
                    outputs=[
                        design_irodori_lora,
                        design_reuse_irodori_lora,
                        clone_irodori_lora,
                        clone_lora_adapter_status,
                    ],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                clone_lora_adapter_open_dir.click(
                    fn=_open_lora_root,
                    inputs=[],
                    outputs=[clone_lora_adapter_status],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                clone_history_refresh.click(
                    fn=_refresh_voice_design_history,
                    inputs=[],
                    outputs=[clone_history],
                    show_progress=False,
                )
                clone_irodori_lora_refresh.click(
                    fn=_lora_adapter_dropdown_update,
                    inputs=[],
                    outputs=[clone_irodori_lora],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                clone_history_delete.click(
                    fn=_delete_voice_design_history_single,
                    inputs=[clone_history],
                    outputs=[clone_history, clone_history_status],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )

            with gr.Tab("高精度クローン") as hifi_tab:
                gr.Markdown(
                    "参照音声と、その音声で話している内容の文字起こしを使います。"
                    "事前に文字起こししたテキストを貼り付けても使えます。"
                    "このモードでは不要な読み上げ混入を防ぐため、英語の制御文は先頭に追加しません。"
                )
                hifi_irodori_notice = gr.Markdown(
                    "選択中のエンジンは高精度クローンには対応していません。Irodori-TTSやQwen3-TTSを使う場合は「声のクローン」タブで参照音声を指定してください。",
                    visible=False,
                )
                with gr.Group() as hifi_voxcpm_group:
                    with gr.Row():
                        with gr.Column():
                            hifi_ref = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="参照音声（アップロード / マイク録音）",
                            )
                            hifi_history = gr.Dropdown(
                                choices=_list_voice_design_history(),
                                value=None,
                                label="声のデザイン履歴から選択（任意）",
                                info="参照音声をアップロードしていない場合、この履歴の声を使います。",
                            )
                            hifi_history_refresh = gr.Button("履歴を更新", variant="secondary", size="sm")
                            _, hifi_recording_script = _add_reference_recording_guide(open_default=True)
                            hifi_script_to_prompt_btn = gr.Button("録音原稿を文字起こし欄へ入れる", variant="secondary")
                            hifi_language = _language_dropdown()
                            hifi_prompt_text = gr.Textbox(
                                value="",
                                label="参照音声の文字起こし（手入力・貼り付け可）",
                                placeholder="参照音声で実際に話している内容を入力してください。例: こんにちは。今日はVoxCPMのテストをしています。",
                                lines=5,
                            )
                            hifi_transcribe_btn = gr.Button("自動文字起こしを試す", variant="secondary")
                            hifi_transcribe_status = gr.Markdown(
                                "自動文字起こしは補助機能です。うまくいかない場合は、上の欄に事前の文字起こしを貼り付けてください。"
                            )
                            hifi_prevent_leading_mix = gr.Checkbox(
                                value=True,
                                label="冒頭の不要な言葉を防ぐ（推奨）",
                                info="有効時は文字起こしを連続生成に使わず、参照音声の声質だけで読み上げます。英語混入が出る場合はこちらを使ってください。",
                            )
                            hifi_intonation = gr.State("")
                            hifi_word_accent = gr.State("")
                            hifi_text = gr.Textbox(
                                value="これは高精度クローンを使った音声生成テストです。",
                                label="続けて読み上げるテキスト",
                                lines=5,
                            )
                            hifi_filename = gr.Textbox(
                                value="",
                                label="保存ファイル名（任意）",
                                placeholder="例: high_fidelity_clone_sample",
                                lines=1,
                            )
                            _add_prosody_controls(hifi_text)
                            hifi_denoise, hifi_normalize, hifi_cfg, hifi_steps = _advanced_settings(include_denoise=True)
                            hifi_btn = gr.Button("高精度クローンで生成", variant="primary", size="lg")
                        with gr.Column():
                            hifi_output = gr.Audio(label="生成された音声")
                            hifi_file = gr.File(label="WAVダウンロード", interactive=False)
                            hifi_output_dir = gr.Textbox(
                                value=str(_output_dir()),
                                label="保存先フォルダ",
                                interactive=False,
                            )
                            hifi_history_delete = gr.Button("選択した履歴を削除", variant="stop", size="sm")
                            hifi_history_status = gr.Markdown("")
                            gr.Markdown(
                                "**使い方**\n\n"
                                "1. 参照音声をアップロードするか、声のデザイン履歴から選びます。\n"
                                "2. 参照音声の文字起こしを入力または貼り付けます。自動文字起こしも試せます。\n"
                                "3. 続けて読み上げたい文章を入力して生成します。\n\n"
                                "英語など不要な言葉が冒頭に入る場合は、推奨設定のまま生成してください。"
                                "文字起こしを厳密に使いたい場合だけ、冒頭防止をオフにします。"
                                "読み方の調整は、読み上げテキスト内の記号で行ってください。"
                                "参照音声の文字起こしが間違っていると、生成冒頭に不要な言葉が混ざることがあります。"
                            )

                hifi_transcribe_btn.click(
                    fn=_transcribe_reference,
                    inputs=[hifi_ref, hifi_history],
                    outputs=[hifi_prompt_text, hifi_transcribe_status],
                    show_progress=True,
                    api_name="transcribe_reference",
                )
                hifi_script_to_prompt_btn.click(
                    fn=_copy_recording_script_to_prompt,
                    inputs=[hifi_recording_script],
                    outputs=[hifi_prompt_text, hifi_transcribe_status],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                hifi_history_refresh.click(
                    fn=_refresh_voice_design_history,
                    inputs=[],
                    outputs=[hifi_history],
                    show_progress=False,
                )
                hifi_history_delete.click(
                    fn=_delete_voice_design_history_single,
                    inputs=[hifi_history],
                    outputs=[hifi_history, hifi_history_status],
                    show_progress=False,
                    api_name=None,
                    api_visibility="private",
                )
                hifi_btn.click(
                    fn=_generate_high_fidelity_clone,
                    inputs=[
                        engine_selector,
                        hifi_text,
                        hifi_ref,
                        hifi_history,
                        hifi_prompt_text,
                        hifi_prevent_leading_mix,
                        hifi_intonation,
                        hifi_word_accent,
                        hifi_language,
                        hifi_filename,
                        hifi_cfg,
                        hifi_normalize,
                        hifi_denoise,
                        hifi_steps,
                    ],
                    outputs=[hifi_output, hifi_file],
                    show_progress=True,
                    api_name="high_fidelity_clone",
                )

        engine_visibility_outputs = [
            app_header,
            engine_status,
            design_language,
            design_voice_age_gender_row,
            design_control,
            design_word_accent_group,
            design_prosody_group,
            design_advanced_group,
            design_qwen3_gacha_group,
            design_btn,
            design_qwen3_single_group,
            clone_irodori_profile_group,
            design_irodori_lora_group,
            design_reuse_irodori_lora_group,
            clone_qwen3_ref_text_group,
            clone_qwen3_corpus_group,
            clone_qwen3_corpus_result_group,
            clone_language,
            clone_control,
            clone_word_accent_group,
            clone_prosody_group,
            clone_advanced_group,
            hifi_tab,
            hifi_voxcpm_group,
            hifi_irodori_notice,
        ]

        engine_selector.change(
            fn=_engine_visibility_updates,
            inputs=[engine_selector],
            outputs=engine_visibility_outputs,
            js=_engine_tab_visibility_js,
            show_progress=False,
            api_name=None,
            api_visibility="private",
        )
        for engine_aware_tab in (design_tab, clone_tab, hifi_tab):
            engine_aware_tab.select(
                fn=_engine_visibility_updates,
                inputs=[engine_selector],
                outputs=engine_visibility_outputs,
                show_progress=False,
                api_name=None,
                api_visibility="private",
            )

        output_dir_apply.click(
            fn=_set_output_dir,
            inputs=[output_dir_global],
            outputs=[
                output_dir_global,
                design_output_dir,
                clone_output_dir,
                hifi_output_dir,
                design_history,
                clone_history,
                hifi_history,
                output_dir_status,
            ],
            show_progress=False,
            api_name=None,
            api_visibility="private",
        )
        output_dir_open.click(
            fn=_open_output_dir,
            inputs=[output_dir_global],
            outputs=[output_dir_status],
            show_progress=False,
            api_name=None,
            api_visibility="private",
        )

    return interface

def run_demo(
    server_name: str = "127.0.0.1",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
    load_denoiser: bool = True,
):
    demo = VoxCPMDemo(model_id=model_id, device=device, load_denoiser=load_denoiser)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
        head=_JAPANESE_UI_FIX_HEAD,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host. Use 127.0.0.1 for local-only access or 0.0.0.0 for LAN access.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--no-denoiser",
        action="store_true",
        help="Do not load ZipEnhancer denoiser at model startup",
    )
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
        load_denoiser=not args.no_denoiser,
    )
