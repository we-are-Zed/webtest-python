if "%cd:~-7%"=="scripts" (
    cd ..
)

@REM 实验组名 实验配置数量 实验配置名依次排列
@REM call scripts/main_multi.bat 1 4 allrecipes-drl-2h-new recipetineats-drl-2h-new cnn-drl-2h-new infobae-drl-2h-new
@REM call scripts/main_multi.bat 1 4 allrecipes-drl-2h-new recipetineats-drl-2h-new cnn-drl-2h-new infobae-drl-2h-new
@REM call scripts/main_multi.bat 1 4 allrecipes-drl-2h-recipetineats-update recipetineats-drl-2h-allrecipes-update allrecipes-drl-2h-recipetineats-stop recipetineats-drl-2h-allrecipes-stop
@REM call scripts/main_multi.bat 1 4 allrecipes-drl-2h-allrecipes-stop allrecipes-drl-2h-allrecipes-update recipetineats-drl-2h-recipetineats-stop recipetineats-drl-2h-recipetineats-update
@REM call scripts/main_multi.bat 1 4 cnn-drl-2h-infobae-update infobae-drl-2h-cnn-update cnn-drl-2h-infobae-stop infobae-drl-2h-cnn-stop
@REM call scripts/main_multi.bat 1 4 cnn-drl-2h-cnn-update infobae-drl-2h-infobae-update cnn-drl-2h-cnn-stop infobae-drl-2h-infobae-stop
@REM call scripts/main_multi.bat 1 4 allrecipes-rl-2h-q recipetineats-rl-2h-q cnn-rl-2h-q infobae-rl-2h-q
@REM call scripts/main_multi.bat 1 4 allrecipes-rl-2h-w recipetineats-rl-2h-w cnn-rl-2h-w infobae-rl-2h-w

@REM call scripts/main_multi.bat 2 4 allrecipes-drl-2h-recipetineats-update recipetineats-drl-2h-allrecipes-update allrecipes-drl-2h-recipetineats-stop recipetineats-drl-2h-allrecipes-stop
@REM call scripts/main_multi.bat 2 4 allrecipes-drl-2h-allrecipes-stop allrecipes-drl-2h-allrecipes-update recipetineats-drl-2h-recipetineats-stop recipetineats-drl-2h-recipetineats-update
@REM call scripts/main_multi.bat 2 4 cnn-drl-2h-infobae-update infobae-drl-2h-cnn-update cnn-drl-2h-infobae-stop infobae-drl-2h-cnn-stop
@REM call scripts/main_multi.bat 2 4 cnn-drl-2h-cnn-update infobae-drl-2h-infobae-update cnn-drl-2h-cnn-stop infobae-drl-2h-infobae-stop


@REM call scripts/main_multi.bat 1 4 cnn-drl-2h-new infobae-drl-2h-new cnn-rl-2h-q infobae-rl-2h-q
@REM call scripts/main_multi.bat 1 4 cnn-drl-2h-infobae-update infobae-drl-2h-cnn-update cnn-drl-2h-infobae-stop infobae-drl-2h-cnn-stop

@REM call scripts/main_multi.bat 1 3 gamespot-drl-2h-new gamespot-rl-2h-q gamespot-rl-2h-w
call scripts/main_multi.bat 1 3 github-drl-2h-new github-rl-2h-q github-rl-2h-w
call scripts/main_multi.bat 1 3 eatingwell-drl-2h-new eatingwell-rl-2h-q eatingwell-rl-2h-w


call scripts/main_multi.bat 2 3 gamespot-drl-2h-new gamespot-rl-2h-q gamespot-rl-2h-w
call scripts/main_multi.bat 2 3 github-drl-2h-new github-rl-2h-q github-rl-2h-w
call scripts/main_multi.bat 2 3 eatingwell-drl-2h-new eatingwell-rl-2h-q eatingwell-rl-2h-w



@REM call scripts/main_multi.bat 1 3 gamespot-drl-4h-new gamespot-rl-4h-q gamespot-rl-4h-w
@REM call scripts/main_multi.bat 1 3 github-drl-4h-new github-rl-4h-q github-rl-4h-w
@REM call scripts/main_multi.bat 1 3 eatingwell-drl-4h-new eatingwell-rl-4h-q eatingwell-rl-4h-w
@REM call scripts/main_multi.bat 1 3 toppr-drl-4h-new toppr-rl-4h-q toppr-rl-4h-w