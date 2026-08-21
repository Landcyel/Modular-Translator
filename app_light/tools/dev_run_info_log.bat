@echo off
rem 以 info 日志落盘模式启动开发版（转写诊断对照用）
set TRANSLATOR_INFO_LOG=1
cd /d "D:\ReasonixProjects\TestFletApp\Reasonix_code\app_lite"
start "" "D:\ReasonixProjects\TestFletApp\Reasonix_code\dependencies\venv\Scripts\python.exe" APP.py
