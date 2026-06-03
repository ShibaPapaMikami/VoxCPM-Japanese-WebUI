@echo off
setlocal
cd /d "%~dp0"

echo JP Voice Studio setup
echo.
echo Default: installs VoxCPM2 WebUI dependencies and launches the app.
echo Optional examples:
echo   JPVoiceStudio_Setup.cmd -AllEngines
echo   JPVoiceStudio_Setup.cmd -WithIrodori -WithQwen3
echo   JPVoiceStudio_Setup.cmd -HostAddress 0.0.0.0 -AllowFirewall
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_all_windows.ps1" %*
if errorlevel 1 (
  echo.
  echo JP Voice Studio setup failed.
  pause
)
