@echo off
rem med_assistant portable entry point (policy-immune: plain cmd, no PowerShell)
rem Runs the stdlib-only launcher module against the embedded python runtime.
if exist "%~dp0python\python.exe" call "%~dp0launcher\hide_coding_dirs.bat" >nul 2>&1
"%~dp0python\python.exe" -m medassistant.launcher %*
