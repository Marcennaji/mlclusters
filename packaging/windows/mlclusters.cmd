@echo off
setlocal

REM ========================================================
REM See the mlclusters_env script for full documentation on the
REM environment variables used by MLClusters
REM ========================================================

REM ========================================================
REM Initialization of the installation directory of MLClusters

REM Test is mlclusters_env is present
if not exist "%~dp0mlclusters_env.cmd" goto ERR_PATH_1

REM Initialize MLClusters env variables
call "%~dp0mlclusters_env"

REM Test is MLClusters environment already set up
if not exist "%MLCLUSTERS_PATH%" goto ERR_PATH_2

REM display mpi configuration problems if any
if not "%MLCLUSTERS_MPI_ERROR%". == "". echo %MLCLUSTERS_MPI_ERROR%

REM Test if batch mode from parameters
set MLCLUSTERS_BATCH_MODE=false
for %%i in (%*) do (
    for %%f in ("-h" "-b" "-s" "-v") do if /I "%%~i"=="%%~f" (
        set MLCLUSTERS_BATCH_MODE=true
        goto BREAK_LOOP
    ) 
)
:BREAK_LOOP

if "%MLCLUSTERS_BATCH_MODE%" == "true" if not "%MLCLUSTERS_JAVA_ERROR%". == "". goto ERR_JAVA 
if "%_IS_CONDA%" == "true" if not "%MLCLUSTERS_BATCH_MODE%" == "true" goto ERR_CONDA

REM Set path
set path=%~dp0;%MLCLUSTERS_JAVA_PATH%;%path%
set classpath=%MLCLUSTERS_CLASSPATH%;%classpath%

REM unset local variables
set "MLCLUSTERS_BATCH_MODE="
set "_IS_CONDA="

REM ========================================================
REM Start MLClusters (with or without parameteres)

if %1.==. goto NOPARAMS
if not %1.==. goto PARAMS

REM Start without parameters
:NOPARAMS
if not exist "%MLCLUSTERS_LAST_RUN_DIR%" md "%MLCLUSTERS_LAST_RUN_DIR%"
if not exist "%MLCLUSTERS_LAST_RUN_DIR%" goto PARAMS

%MLCLUSTERS_MPI_COMMAND% "%MLCLUSTERS_PATH%" -o "%MLCLUSTERS_LAST_RUN_DIR%\scenario._kh" -e "%MLCLUSTERS_LAST_RUN_DIR%\log.txt"
if %errorlevel% EQU 0 goto END
goto ERR_RETURN_CODE

REM Start with parameters
:PARAMS
%MLCLUSTERS_MPI_COMMAND% "%MLCLUSTERS_PATH%" %*
if %errorlevel% EQU 0 goto END
goto ERR_RETURN_CODE

REM ========================================================
REM Error messages

:ERR_PATH_1
start "MLCLUSTERS CONFIGURATION PROBLEM" echo ERROR "mlclusters_env.cmd is missing in directory %~dp0"
exit /b 1

:ERR_PATH_2
start "MLCLUSTERS CONFIGURATION PROBLEM" echo ERROR "Incorrect installation directory for MLClusters (File %MLCLUSTERS_PATH% not found)"
exit /b 1

:ERR_RETURN_CODE
start "MLCLUSTERS EXECUTION PROBLEM" cmd /k "echo ERROR MLClusters ended with return code %errorlevel% & echo Contents of the log file at %MLCLUSTERS_LAST_RUN_DIR%\log.txt: & type %MLCLUSTERS_LAST_RUN_DIR%\log.txt"
goto END

:ERR_JAVA
start "MLCLUSTERS CONFIGURATION PROBLEM" echo ERROR "%MLCLUSTERS_JAVA_ERROR%"
exit /b 1

:ERR_CONDA
echo GUI is not available, please use the '-b' flag
exit /b 1

:END
endlocal

REM Return 1 if fatal error, 0 otherwise
exit /b %errorlevel%