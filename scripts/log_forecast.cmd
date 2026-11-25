@echo off
rem Hourly live forecast logger, for Windows Task Scheduler. Writes to the data folder, live subfolder.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m src.live.log_forecasts
