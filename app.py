import os
import re
import sys
import logging
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
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
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


# ---------- Model ----------

def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


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

    def _list_voice_design_history():
        output_dir = Path.cwd() / "outputs"
        if not output_dir.exists():
            return []
        choices = []
        for path in sorted(output_dir.glob("voice_design_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%m/%d %H:%M")
            choices.append((f"{timestamp} - {path.name}", str(path)))
        return choices

    def _save_wav_for_download(sr: int, wav_np: np.ndarray, prefix: str) -> str:
        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{prefix}_{timestamp}_{uuid4().hex[:8]}.wav"
        processing_utils.audio_to_file(sr, np.asarray(wav_np), str(output_path), format="wav")
        logger.info(f"Saved generated WAV for download: {output_path}")
        return str(output_path)

    def _generate_design(
        text: str,
        control_instruction: str,
        intonation_instruction: str,
        word_accent_instruction: str,
        target_language: str,
        cfg_value: float,
        do_normalize: bool,
        dit_steps: int,
    ):
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction=control_instruction,
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
        output_path = _save_wav_for_download(sr, wav_np, "voice_design")
        return (sr, wav_np), output_path, gr.update(choices=_list_voice_design_history(), value=output_path)

    def _refresh_voice_design_history():
        choices = _list_voice_design_history()
        return gr.update(choices=choices, value=choices[0][1] if choices else None)

    def _generate_from_design_history(
        history_wav: Optional[str],
        text: str,
        target_language: str,
        cfg_value: float,
        do_normalize: bool,
        dit_steps: int,
    ):
        if not history_wav:
            raise ValueError("再利用する声を履歴から選んでください。")
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
        return (sr, wav_np), _save_wav_for_download(sr, wav_np, "voice_design_reuse")

    def _generate_clone(
        text: str,
        control_instruction: str,
        intonation_instruction: str,
        word_accent_instruction: str,
        ref_wav: Optional[str],
        target_language: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
    ):
        if not ref_wav:
            raise ValueError("声のクローンには参照音声を指定してください。")
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction=control_instruction,
            intonation_instruction=intonation_instruction,
            word_accent_instruction=word_accent_instruction,
            target_language=target_language,
            reference_wav_path_input=ref_wav,
            prompt_text="",
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
        )
        return (sr, wav_np), _save_wav_for_download(sr, wav_np, "voice_clone")

    def _generate_high_fidelity_clone(
        text: str,
        ref_wav: Optional[str],
        prompt_text_value: str,
        intonation_instruction: str,
        word_accent_instruction: str,
        target_language: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
    ):
        if not ref_wav:
            raise ValueError("高精度クローンには参照音声を指定してください。")
        if not (prompt_text_value or "").strip():
            raise ValueError("高精度クローンには参照音声の文字起こしが必要です。")
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction="",
            intonation_instruction=intonation_instruction,
            word_accent_instruction=word_accent_instruction,
            target_language=target_language,
            reference_wav_path_input=ref_wav,
            prompt_text=prompt_text_value,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
        )
        return (sr, wav_np), _save_wav_for_download(sr, wav_np, "high_fidelity_clone")

    def _transcribe_reference(audio_path: Optional[str]):
        if not audio_path:
            return gr.update(), "参照音声を指定してから、自動文字起こしを試してください。"
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
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

    with gr.Blocks(title="JPVoxCPM WebUI") as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        gr.Markdown("## JPVoxCPM WebUI\n**日本語で使いやすい VoxCPM2 音声生成・声クローンWeb UIです。**")
        gr.Markdown("**用途に合わせてモードを選んでください。** 各画面には、その生成方法に必要な入力だけを表示しています。")

        with gr.Tabs():
            with gr.Tab("声のデザイン"):
                gr.Markdown(
                    "参照音声を使わず、声の雰囲気を文章で指定して新しい声を作ります。"
                    "男性声・女性声・話す速さなどの日本語指定は、内部でモデル向けの声質タグに補強されます。"
                )
                with gr.Row():
                    with gr.Column():
                        design_language = _language_dropdown()
                        design_control = gr.Textbox(
                            value="低めの落ち着いた日本語の男性ナレーション。大人の男性声で、聞き取りやすく、少しゆっくり話す。",
                            label="声の指示",
                            placeholder="例: 低めの男性ナレーション / やさしい女性の声 / 元気なキャラクター声",
                            lines=3,
                        )
                        design_intonation = gr.State("")
                        design_word_accent = _add_word_accent_controls()
                        design_text = gr.Textbox(
                            value=DEFAULT_TARGET_TEXT,
                            label="読み上げテキスト",
                            lines=5,
                        )
                        _add_prosody_controls(design_text)
                        _, design_normalize, design_cfg, design_steps = _advanced_settings(
                            include_denoise=False,
                            cfg_default=2.6,
                        )
                        design_btn = gr.Button("この声を生成", variant="primary", size="lg")
                    with gr.Column():
                        design_output = gr.Audio(label="生成された音声")
                        design_file = gr.File(label="WAVダウンロード", interactive=False)
                        gr.Markdown(
                            "**使い方**\n\n"
                            "1. 声の指示に、声質や話し方を書きます。\n"
                            "2. 読み上げたい文章を入力します。\n"
                            "3. 「この声を生成」を押します。"
                        )
                        with gr.Accordion("声のデザイン履歴から再利用", open=True):
                            gr.Markdown(
                                "声のデザインで生成した音声を参照音声として使い、"
                                "同じ声質に近い声で別のセリフを生成します。"
                            )
                            design_history = gr.Dropdown(
                                choices=_list_voice_design_history(),
                                value=(_list_voice_design_history()[0][1] if _list_voice_design_history() else None),
                                label="再利用する声",
                                info="新しく声を生成すると、この履歴に追加されます。",
                            )
                            design_history_refresh = gr.Button("履歴を更新", variant="secondary", size="sm")
                            design_reuse_text = gr.Textbox(
                                value="この声で、別のセリフも読んでみます。",
                                label="この声で読み上げるテキスト",
                                lines=4,
                            )
                            design_reuse_btn = gr.Button("履歴の声で生成", variant="primary")
                            design_reuse_output = gr.Audio(label="履歴の声で生成された音声")
                            design_reuse_file = gr.File(label="WAVダウンロード", interactive=False)

                design_btn.click(
                    fn=_generate_design,
                    inputs=[
                        design_text,
                        design_control,
                        design_intonation,
                        design_word_accent,
                        design_language,
                        design_cfg,
                        design_normalize,
                        design_steps,
                    ],
                    outputs=[design_output, design_file, design_history],
                    show_progress=True,
                    api_name="design",
                )
                design_history_refresh.click(
                    fn=_refresh_voice_design_history,
                    inputs=[],
                    outputs=[design_history],
                    show_progress=False,
                )
                design_reuse_btn.click(
                    fn=_generate_from_design_history,
                    inputs=[
                        design_history,
                        design_reuse_text,
                        design_language,
                        design_cfg,
                        design_normalize,
                        design_steps,
                    ],
                    outputs=[design_reuse_output, design_reuse_file],
                    show_progress=True,
                    api_name="design_reuse",
                )

            with gr.Tab("声のクローン"):
                gr.Markdown("参照音声の声質をもとに、別の文章を読み上げます。声の指示で雰囲気の調整もできます。")
                with gr.Row():
                    with gr.Column():
                        clone_ref = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="参照音声",
                        )
                        clone_language = _language_dropdown()
                        clone_control = gr.Textbox(
                            value="自然で聞き取りやすく、落ち着いた話し方",
                            label="声の指示（任意）",
                            placeholder="例: 少し明るく / ゆっくり丁寧に / 感情を抑えて",
                            lines=2,
                        )
                        clone_intonation = gr.State("")
                        clone_word_accent = _add_word_accent_controls()
                        clone_text = gr.Textbox(
                            value="これは参照音声を使った声のクローン生成テストです。",
                            label="読み上げテキスト",
                            lines=5,
                        )
                        _add_prosody_controls(clone_text)
                        clone_denoise, clone_normalize, clone_cfg, clone_steps = _advanced_settings(include_denoise=True)
                        clone_btn = gr.Button("この声で生成", variant="primary", size="lg")
                    with gr.Column():
                        clone_output = gr.Audio(label="生成された音声")
                        clone_file = gr.File(label="WAVダウンロード", interactive=False)
                        gr.Markdown(
                            "**使い方**\n\n"
                            "1. クローンしたい声の音声をアップロードします。\n"
                            "2. 必要なら声の指示で雰囲気を調整します。\n"
                            "3. 読み上げテキストを入力して生成します。"
                        )

                clone_btn.click(
                    fn=_generate_clone,
                    inputs=[
                        clone_text,
                        clone_control,
                        clone_intonation,
                        clone_word_accent,
                        clone_ref,
                        clone_language,
                        clone_cfg,
                        clone_normalize,
                        clone_denoise,
                        clone_steps,
                    ],
                    outputs=[clone_output, clone_file],
                    show_progress=True,
                    api_name="clone",
                )

            with gr.Tab("高精度クローン"):
                gr.Markdown(
                    "参照音声と、その音声で話している内容の文字起こしを使います。"
                    "事前に文字起こししたテキストを貼り付けても使えます。"
                    "このモードでは不要な読み上げ混入を防ぐため、英語の制御文は先頭に追加しません。"
                )
                with gr.Row():
                    with gr.Column():
                        hifi_ref = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="参照音声",
                        )
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
                        hifi_intonation = gr.State("")
                        hifi_word_accent = gr.State("")
                        hifi_text = gr.Textbox(
                            value="これは高精度クローンを使った音声生成テストです。",
                            label="続けて読み上げるテキスト",
                            lines=5,
                        )
                        _add_prosody_controls(hifi_text)
                        hifi_denoise, hifi_normalize, hifi_cfg, hifi_steps = _advanced_settings(include_denoise=True)
                        hifi_btn = gr.Button("高精度クローンで生成", variant="primary", size="lg")
                    with gr.Column():
                        hifi_output = gr.Audio(label="生成された音声")
                        hifi_file = gr.File(label="WAVダウンロード", interactive=False)
                        gr.Markdown(
                            "**使い方**\n\n"
                            "1. 参照音声をアップロードします。\n"
                            "2. 参照音声の文字起こしを入力または貼り付けます。自動文字起こしも試せます。\n"
                            "3. 続けて読み上げたい文章を入力して生成します。\n\n"
                            "このモードでは声の指示は使わず、参照音声と文字起こしを優先します。"
                            "読み方の調整は、読み上げテキスト内の記号で行ってください。"
                            "参照音声の文字起こしが間違っていると、生成冒頭に不要な言葉が混ざることがあります。"
                        )

                hifi_transcribe_btn.click(
                    fn=_transcribe_reference,
                    inputs=[hifi_ref],
                    outputs=[hifi_prompt_text, hifi_transcribe_status],
                    show_progress=True,
                    api_name="transcribe_reference",
                )
                hifi_btn.click(
                    fn=_generate_high_fidelity_clone,
                    inputs=[
                        hifi_text,
                        hifi_ref,
                        hifi_prompt_text,
                        hifi_intonation,
                        hifi_word_accent,
                        hifi_language,
                        hifi_cfg,
                        hifi_normalize,
                        hifi_denoise,
                        hifi_steps,
                    ],
                    outputs=[hifi_output, hifi_file],
                    show_progress=True,
                    api_name="high_fidelity_clone",
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
