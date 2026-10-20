@echo off
rem Idempotent: re-hide everything except Start.bat and README.txt after
rem unpacking so Explorer extracts give the doctor a one-click folder.
rem Covers launcher, data, models, logs - .env and dist-manifest.json too;
rem logs exists only after the first run. Safe to run repeatedly.
rem Invoked by Start.bat on every run.
pushd "%~dp0.."
for %%D in (python engine medassistant data launcher logs models) do attrib +h "%%D" /d 2>nul
for %%F in (.env dist-manifest.json) do attrib +h "%%F" 2>nul
popd
