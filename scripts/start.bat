if "%cd:~-7%"=="scripts" (
    cd ..
)
@REM 实验组名 实验配置名
call scripts/main.bat 1 recipetineats-drl-2h-new
call scripts/main.bat 1 allrecipes-drl-2h-new
call scripts/main.bat 1 recipetineats-drl-2h-allrecipes-update
call scripts/main.bat 1 allrecipes-drl-2h-recipetineats-update
call scripts/main.bat 1 recipetineats-drl-2h-allrecipes-stop
call scripts/main.bat 1 allrecipes-drl-2h-recipetineats-stop
call scripts/main.bat 1 recipetineats-rl-2h-q
call scripts/main.bat 1 recipetineats-rl-2h-w
call scripts/main.bat 1 allrecipes-rl-2h-q
call scripts/main.bat 1 allrecipes-rl-2h-w
