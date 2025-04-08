%REM We assume the package assets were downloaded to packaging\windows\nsis\assets
%REM Before building the installer, update the following script arguments, mainly:
%REM - MLCLUSTERS_VERSION: MLClusters version for the installer.
%REM                   Should be identical to the value that is set in MLCLUSTERS_STR in src/Learning/KWUtils/KWMLClustersVersion.h
%REM - MLCLUSTERS_REDUCED_VERSION: MLClusters version without suffix and only digits and periods.
%REM                           Thus, the pre-release fields of MLCLUSTERS_VERSION are ignored in MLCLUSTERS_REDUCED_VERSION.

makensis ^
   /DMLCLUSTERS_VERSION=10.7.0-b.0 ^
   /DMLCLUSTERS_REDUCED_VERSION=10.7.0 ^
   /DMLCLUSTERS_WINDOWS_BUILD_DIR=..\..\..\build ^
   /DJRE_PATH=.\assets\jre\ ^
   /DMSMPI_INSTALLER_PATH=.\assets\msmpisetup.exe ^
   /DMSMPI_VERSION=10.1.3 ^
   /DMLCLUSTERS_SAMPLES_DIR=.\assets\samples ^
   /DMLCLUSTERS_DOC_DIR=.\assets\doc ^
   mlclusters.nsi