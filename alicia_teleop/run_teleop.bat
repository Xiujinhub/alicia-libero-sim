@echo off
REM ============================================================
REM  Alicia-D virtual teleoperation launcher  (conda env: lerobot)
REM  NOTE: keep this file ASCII-only - cmd.exe reads .bat as ANSI/GBK
REM        and non-ASCII bytes can break the command line.
REM
REM  Usage:
REM    run_teleop.bat                        leader arm, auto-detect serial port
REM    run_teleop.bat list                   list serial ports
REM    run_teleop.bat probe                  read-only leader probe (port check)
REM    run_teleop.bat virtual                virtual leader (keyboard, no hardware)
REM    run_teleop.bat virtual --duration 10  self test, exit after 10 seconds
REM    run_teleop.bat --port COM5 --disable-torque
REM    run_teleop.bat --help
REM ============================================================
setlocal
set "ENV_NAME=lerobot"
set "HERE=%~dp0"
set "PY=%HERE%alicia_virtual_teleop.py"

if "%~1"=="" (
    call conda run --no-capture-output -n %ENV_NAME% python "%PY%" --source leader
) else if /I "%~1"=="virtual" (
    call conda run --no-capture-output -n %ENV_NAME% python "%PY%" --source virtual %2 %3 %4 %5 %6 %7
) else if /I "%~1"=="auto" (
    call conda run --no-capture-output -n %ENV_NAME% python "%PY%" --source auto %2 %3 %4 %5 %6 %7
) else if /I "%~1"=="list" (
    call conda run --no-capture-output -n %ENV_NAME% python "%PY%" --list-ports
) else if /I "%~1"=="probe" (
    call conda run --no-capture-output -n %ENV_NAME% python "%HERE%alicia_leader_probe.py" %2 %3 %4 %5 %6 %7
) else (
    call conda run --no-capture-output -n %ENV_NAME% python "%PY%" %*
)
if errorlevel 1 pause
endlocal
