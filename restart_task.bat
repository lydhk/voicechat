@echo off
set TASK_NAME=VoiceChatMain

echo Stopping task "%TASK_NAME%"...
schtasks /End /TN "%TASK_NAME%"

echo Waiting for process to exit...
timeout /t 3 /nobreak >nul

echo Starting task "%TASK_NAME%"...
schtasks /Run /TN "%TASK_NAME%"

echo Done.
pause
