@echo off
setlocal enabledelayedexpansion

taskkill /f /im chrome.exe /t
taskkill /f /im chromedriver.exe /t

set session=%1
set total_loop=%2
set loop_round=1

for /L %%i in (1, 1, %total_loop%) do (
    if exist agent%%i_running (
        del agent%%i_running
    )
)

echo Running %total_loop% settings...
:loop
if "%3"=="" goto :next_part
echo Running %3...
type nul > agent%loop_round%_running
start cmd /c "conda activate webtest && python ./main.py --profile=%3 --session=%session% && del agent%loop_round%_running"
shift
set /a loop_round=%loop_round%+1
goto :loop

:next_part
call scripts/CheckFinish_multi.bat %total_loop%
timeout /t 5 /nobreak >nul

taskkill /f /im chrome.exe /t
taskkill /f /im chromedriver.exe /t

timeout /t 5 /nobreak >nul

echo Copying files...

if not exist ExpData mkdir ExpData
@REM move "webtest_output\result" "ExpData"

endlocal