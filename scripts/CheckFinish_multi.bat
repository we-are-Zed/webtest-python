@echo off
setlocal

echo Waiting for agents to finish...

:wait
for /L %%i in (1, 1, %1) do (
    if exist agent%%i_running (
        timeout /t 1 /nobreak >nul
        goto wait
    )
)

endlocal
