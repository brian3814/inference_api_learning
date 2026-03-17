@echo off

rem Load root .env if it exists
if exist "%~dp0.env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0.env") do (
        set "%%A=%%B"
    )
)

cd /d "%~dp0backend"
uv run python run.py
