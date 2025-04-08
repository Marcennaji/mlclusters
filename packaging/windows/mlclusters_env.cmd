@echo off

if %1.==--env. goto DISPLAY_ENV
if %*.==. goto SET_ENV

:HELP
echo Usage: mlclusters_env [-h, --help] [--env]
echo mlclusters_env is an internal script intended to be used by MLClusters tool and MLClusters wrappers only.
echo It sets all the environment variables required by the MLClusters to run correctly (Java, MPI, etc).
echo Options:
echo    -h, --help show this help message and exit
echo    -env show the environment list and exit
echo.
echo The following variables are used to set the path and classpath for the prerequisite of MLClusters.
echo.
echo MLCLUSTERS_PATH: full path of MLClusters executable
echo MLCLUSTERS_MPI_COMMAND: MPI command to call the MLClusters tool
echo MLCLUSTERS_JAVA_PATH: path of Java tool, to add in path
echo MLCLUSTERS_CLASSPATH: MLClusters java libraries, to add in classpath
echo MLCLUSTERS_DRIVERS_PATH: search path of the drivers (by default MLClusters bin directory if not defined)
echo.
echo If they are not already defined, the following variables used by MLClusters are set:
echo.
echo MLCLUSTERS_LAST_RUN_DIR: directory where MLClusters writes output command file and log
echo   (when not defined with -e and -o)
echo MLCLUSTERS_PROC_NUMBER: processes number launched by MLClusters (it's default value corresponds to the
echo   number of physical cores of the computer)
echo.
echo The following variables are not defined by default and can be used to change some default
echo  properties of MLClusters:
echo.
echo MLCLUSTERS_TMP_DIR: MLClusters temporary directory location (default: the system default).
echo   This location can be modified from the tool as well.
echo MLCLUSTERS_MEMORY_LIMIT: MLClusters memory limit in MB (default: the system memory limit).
echo   The minimum value is 100 MB; this setting is ignored if it is above the system's memory limit.
echo   It can only be reduced from the tool.
echo MLCLUSTERS_API_MODE: standard or api mode for the management of output result files created by MLClusters
echo   In standard mode, the result files are stored in the train database directory,
echo   unless an absolute path is specified, and the file extension is forced if necessary.
echo   In api mode, the result files are stored in the current working directory, using the specified results as is.
echo   . default behavior if not set: standard mode
echo   . set to 'false' to force standard mode
echo   . set to 'true' to force api mode
echo MLCLUSTERS_RAW_GUI: graphical user interface for file name selection
echo   . default behavior if not set: depending on the file drivers available for MLClusters
echo   . set to 'true' to allow file name selection with uri schemas
echo   . set to 'false' to allow local file name selection only with a file selection dialog
echo.
echo In case of configuration problems, the variables MLCLUSTERS_JAVA_ERROR and MLCLUSTERS_MPI_ERROR contain error messages.

if not %2.==. exit /b 1
if %1.==-h. exit /b 0
if %1.==--help. exit /b 0
exit /b 1

REM Set MLClusters environment variables
:DISPLAY_ENV
setlocal
set DISPLAY_ENV=true

:SET_ENV
REM Initialize exported variables
set "MLCLUSTERS_PATH="
set "MLCLUSTERS_MPI_COMMAND="
set "MLCLUSTERS_JAVA_PATH="
set "MLCLUSTERS_CLASSPATH="
set "MLCLUSTERS_JAVA_ERROR="
set "MLCLUSTERS_MPI_ERROR="

REM Set MLClusters home to parent directory
for %%a in ("%~dp0..") do set "_MLCLUSTERS_HOME=%%~fa"

REM MLCLUSTERS_PATH
set "MLCLUSTERS_PATH=%_MLCLUSTERS_HOME%\bin\mlclusters.exe"

REM MLCLUSTERS_LAST_RUN_DIR
if "%MLCLUSTERS_LAST_RUN_DIR%". == "". set "MLCLUSTERS_LAST_RUN_DIR=%USERPROFILE%\mlclusters_data\lastrun"

REM MLCLUSTERS_PROC_NUMBER
if "%MLCLUSTERS_PROC_NUMBER%". == "". for /f %%i in ('"%~dp0_khiopsgetprocnumber"') do set "MLCLUSTERS_PROC_NUMBER=%%i"
if "%MLCLUSTERS_PROC_NUMBER%". == "". set "MLCLUSTERS_PROC_NUMBER=1"

REM Set MPI binary (mpiexec)
if %MLCLUSTERS_PROC_NUMBER% LEQ 2 goto MPI_DONE
goto SET_MPI_SYSTEM_WIDE

:MPI_PARAMS
REM Add the MPI parameters
if not "%MLCLUSTERS_MPI_COMMAND%." == "." set "MLCLUSTERS_MPI_COMMAND="%MLCLUSTERS_MPI_COMMAND%" -n %MLCLUSTERS_PROC_NUMBER%"
:MPI_DONE

set _MLCLUSTERS_GUI=
if "%_MLCLUSTERS_GUI%" == "false" GOTO SKIP_GUI

REM Set Java environment
set _JAVA_ERROR=false
if not exist "%_MLCLUSTERS_HOME%\jre\bin\server\" set _JAVA_ERROR=true
if not exist "%_MLCLUSTERS_HOME%\jre\bin\" set _JAVA_ERROR=true

if  "%_JAVA_ERROR%" == "false" (
    set "MLCLUSTERS_JAVA_PATH=%_MLCLUSTERS_HOME%\jre\bin\server\;%_MLCLUSTERS_HOME%\jre\bin\"
) else set "MLCLUSTERS_JAVA_ERROR=The JRE is missing in MLClusters home directory, please reinstall MLClusters"

REM MLCLUSTERS_CLASSPATH
set "MLCLUSTERS_CLASSPATH=%_MLCLUSTERS_HOME%\bin\norm.jar"
set "MLCLUSTERS_CLASSPATH=%_MLCLUSTERS_HOME%\bin\khiops.jar;%MLCLUSTERS_CLASSPATH%"

:SKIP_GUI





REM unset local variables
set "_MLCLUSTERS_GUI="
set "_JAVA_ERROR="
set "_MLCLUSTERS_HOME="

if not "%DISPLAY_ENV%" == "true" exit /b 0

REM Print the environment list on the standard output
echo MLCLUSTERS_PATH %MLCLUSTERS_PATH%
echo MLCLUSTERS_MPI_COMMAND %MLCLUSTERS_MPI_COMMAND%
echo MLCLUSTERS_JAVA_PATH %MLCLUSTERS_JAVA_PATH%
echo MLCLUSTERS_CLASSPATH %MLCLUSTERS_CLASSPATH%
echo MLCLUSTERS_LAST_RUN_DIR %MLCLUSTERS_LAST_RUN_DIR%
echo MLCLUSTERS_PROC_NUMBER %MLCLUSTERS_PROC_NUMBER%
echo MLCLUSTERS_TMP_DIR %MLCLUSTERS_TMP_DIR%
echo MLCLUSTERS_MEMORY_LIMIT %MLCLUSTERS_MEMORY_LIMIT%
echo MLCLUSTERS_API_MODE %MLCLUSTERS_API_MODE%
echo MLCLUSTERS_RAW_GUI %MLCLUSTERS_RAW_GUI%
echo MLCLUSTERS_DRIVERS_PATH %MLCLUSTERS_DRIVERS_PATH%
echo MLCLUSTERS_JAVA_ERROR %MLCLUSTERS_JAVA_ERROR%
echo MLCLUSTERS_MPI_ERROR %MLCLUSTERS_MPI_ERROR%
endlocal
exit /b 0

REM Set mpiexec path for conda installation
:SET_MPI_CONDA
set "MLCLUSTERS_MPI_COMMAND=%_MLCLUSTERS_HOME%\Library\bin\mpiexec.exe"
if not exist "%MLCLUSTERS_MPI_COMMAND%" goto ERR_MPI
goto MPI_PARAMS

REM Set mpiexec path for system wide installation
:SET_MPI_SYSTEM_WIDE
REM ... with the standard variable MSMPI_BIN
set "MLCLUSTERS_MPI_COMMAND=%MSMPI_BIN%mpiexec.exe"
if  exist "%MLCLUSTERS_MPI_COMMAND%" goto MPI_PARAMS
REM ... if MSMPI_BIN is not correctly defined 
REM ... we try to call directly mpiexec (assuming its path is in the 'path' variable)
set "MLCLUSTERS_MPI_COMMAND=mpiexec"
where /q "%MLCLUSTERS_MPI_COMMAND%"
if ERRORLEVEL 1 goto ERR_MPI
REM ... finally we check if it is the good MPI implementation: "Microsoft MPI"
"%MLCLUSTERS_MPI_COMMAND%" | findstr /c:"Microsoft MPI" > nul
if ERRORLEVEL 1 goto ERR_MPI_IMPL
goto MPI_PARAMS


:ERR_MPI
set "MLCLUSTERS_MPI_ERROR=We didn't find mpiexec in the regular path. Parallel computation is unavailable: MLClusters is launched in serial"
set "MLCLUSTERS_MPI_COMMAND="
goto MPI_DONE

:ERR_MPI_IMPL
set "MLCLUSTERS_MPI_ERROR=We can't find the right implementation of mpiexec, we expect to find Microsoft MPI. Parallel computation is unavailable: MLClusters is launched in serial"
set "MLCLUSTERS_MPI_COMMAND="
goto MPI_DONE
