@echo off
REM Double-click to open the default model (Alicia_D_v5_6_gripper_100mm).
REM Usage: synria_sim.bat [model-name] [simulate^|studio^|python]
REM Examples: synria_sim.bat Alicia_M_v1_2_follower
REM           synria_sim.bat Alicia_D_v5_6_gripper_50mm python
setlocal
set "MODEL=%~1"
set "VIEWER=%~2"
if "%VIEWER%"=="" set "VIEWER=simulate"

if "%MODEL%"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0synria_sim.ps1" -Viewer %VIEWER%
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0synria_sim.ps1" -Model "%MODEL%" -Viewer %VIEWER%
)
if errorlevel 1 pause
endlocal
