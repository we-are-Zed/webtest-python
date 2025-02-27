@echo off
setlocal

echo Waiting for webtest to finish...

:wait
if exist webtest_running (
    timeout /t 1 /nobreak >nul
    goto wait
)

endlocal
