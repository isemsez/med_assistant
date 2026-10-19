@echo off
rem Idempotent: re-hide the coding directories (python\, engine\, medassistant\)
rem so Explorer/.NET extracts don't expose internals. Safe to run repeatedly.
rem Also invoked by Start.bat on first run.
attrib +h "python"  /d 2>nul
attrib +h "engine"  /d 2>nul
attrib +h "medassistant" /d 2>nul
