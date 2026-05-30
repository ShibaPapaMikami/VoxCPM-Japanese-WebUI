@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch_webui.ps1" %*
if errorlevel 1 (
  echo.
  echo VoxCPM Web UI failed to start.
  pause
)
