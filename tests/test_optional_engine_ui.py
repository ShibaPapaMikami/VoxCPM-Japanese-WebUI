from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import app


class OptionalEngineUITests(unittest.TestCase):
    def test_optional_engine_missing_status_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_irodori = Path(tmp_dir) / "missing_irodori"
            demo = app.VoxCPMDemo(device="cpu", load_denoiser=False)

            with (
                mock.patch.object(demo, "irodori_project_dir", return_value=missing_irodori),
                mock.patch.object(app.VoxCPMDemo, "qwen3_package_available", staticmethod(lambda: False)),
            ):
                irodori_status = demo.irodori_status()
                qwen3_status = demo.qwen3_status()

        self.assertIn("Irodori-TTSは未セットアップです", irodori_status)
        self.assertIn("scripts\\setup_irodori_tts.ps1", irodori_status)
        self.assertIn("次にすること:", irodori_status)
        self.assertIn("Qwen3-TTSは未セットアップです", qwen3_status)
        self.assertIn("scripts\\setup_qwen3_tts.ps1", qwen3_status)
        self.assertIn("次にすること:", qwen3_status)

    def test_next_action_helper_formats_button_and_command(self) -> None:
        message = app._with_next_action(
            "テストエラーです。",
            button="生成ボタンを押してください。",
            command="powershell -ExecutionPolicy Bypass -File scripts\\check_setup.ps1",
            after="Web UIを再起動してください。",
        )

        self.assertIn("テストエラーです。", message)
        self.assertIn("次にすること:", message)
        self.assertIn("- 画面: 生成ボタンを押してください。", message)
        self.assertIn("- コマンド: `powershell -ExecutionPolicy Bypass -File scripts\\check_setup.ps1`", message)
        self.assertIn("- その後: Web UIを再起動してください。", message)

    def test_ui_builds_when_optional_engines_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_irodori = Path(tmp_dir) / "missing_irodori"
            with (
                mock.patch.object(app.VoxCPMDemo, "irodori_project_dir", return_value=missing_irodori),
                mock.patch.object(app.VoxCPMDemo, "qwen3_package_available", staticmethod(lambda: False)),
            ):
                demo = app.VoxCPMDemo(device="cpu", load_denoiser=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    interface = app.create_demo_interface(demo)

        try:
            self.assertIsNotNone(interface)
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                interface.close()


if __name__ == "__main__":
    unittest.main()
