@echo off
rem Run the project unit suite (unittest, no pytest needed) from the project root.
rem Output goes to unittest_run.txt in this folder.
cd /d "%~dp0"
python -m unittest discover tests > unittest_run.txt 2>&1
