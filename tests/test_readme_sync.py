from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_FILES = [
    ROOT / "README.md",
    ROOT / "README_en.md",
    ROOT / "README_zh.md",
]


class ReadmeSyncTests(unittest.TestCase):
    def test_key_public_release_topics_are_present_in_all_languages(self) -> None:
        required_terms = [
            "Irodori-TTS",
            "Qwen3-TTS",
            "Voice-Design-Cloner",
            "Style-Bert-VITS2",
            "esd.list",
            "README_SETUP_JA.md",
            "docs/SMOKE_TEST_JA.md",
            "docs/SAMPLES_JA.md",
            "docs/GITHUB_RELEASE_JA.md",
            "docs/ROADMAP_JA.md",
            "THIRD_PARTY_NOTICES.md",
        ]

        for readme_path in README_FILES:
            with self.subTest(readme=readme_path.name):
                text = readme_path.read_text(encoding="utf-8")
                for term in required_terms:
                    self.assertIn(term, text)

    def test_language_navigation_is_present_in_all_readmes(self) -> None:
        for readme_path in README_FILES:
            with self.subTest(readme=readme_path.name):
                text = readme_path.read_text(encoding="utf-8")
                self.assertIn("日本語", text)
                self.assertIn("English", text)
                self.assertIn("中文", text)


if __name__ == "__main__":
    unittest.main()
