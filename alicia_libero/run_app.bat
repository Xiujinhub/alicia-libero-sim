@echo off
REM ============================================================
REM  Alicia-D x LIBERO desktop manipulation studio (PySide6)
REM  NOTE: keep ASCII only - cmd.exe reads .bat as ANSI/GBK.
REM
REM  Usage:
REM    run_app.bat            launch the UI (conda env: lerobot)
REM    run_app.bat catalog    regenerate assets_catalog.json only
REM    run_app.bat scenes     regenerate all task scene XMLs only
REM ============================================================
setlocal
set "ENV_NAME=lerobot"
set "HERE=%~dp0"

if /I "%~1"=="catalog" (
    call conda run --no-capture-output -n %ENV_NAME% python "%HERE%libero_catalog.py" --table
    goto :done
)
if /I "%~1"=="scenes" (
    call conda run --no-capture-output -n %ENV_NAME% python "%HERE%build_all_scenes.py"
    goto :done
)

call conda run --no-capture-output -n %ENV_NAME% python "%HERE%alicia_libero_app.py"
if errorlevel 1 pause
:done
endlocal
