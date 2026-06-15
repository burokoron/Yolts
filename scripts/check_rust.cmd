@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_rust.ps1"

