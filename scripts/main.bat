@echo off
setlocal enabledelayedexpansion

for %%p in (%2) do (
        echo Running %%p...

        taskkill /f /im chrome.exe /t
        taskkill /f /im chromedriver.exe /t

        if exist webtest_running (
            del webtest_running
        )

        type nul > webtest_running
        start cmd /c "conda activate webtest && python ./main.py --profile=%%p --session=%1 && del webtest_running"

        call scripts/CheckFinish.bat %%n

        timeout /t 5 /nobreak >nul

        taskkill /f /im chrome.exe /t
        taskkill /f /im chromedriver.exe /t

        timeout /t 5 /nobreak >nul

@REM         echo Copying files...
@REM
@REM         set CURRENT_DATE=!DATE:/=_!
@REM         set CURRENT_DATE=!CURRENT_DATE: =_!
@REM         set CURRENT_TIME=!TIME::=_!
@REM         set CURRENT_TIME=!CURRENT_TIME: =_!
@REM         set CURRENT_TIME=!CURRENT_TIME:.=_!
@REM
@REM         if not exist ExpData mkdir ExpData
@REM         move "webtest_output\result" "ExpData\%3"
)

endlocal