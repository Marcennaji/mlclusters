# MLClusters NSIS packaging
This folder contains the scripts to generate the MLClusters Windows installer. It is built with
[NSIS](https://nsis.sourceforge.io/Download). See also the [Release Process wiki
page](https://github.com/MLClustersML/mlclusters/wiki/Release-Process).

## What the installer does
Besides installing the MLClusters executables, the installer automatically detects the presence of:
- [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)

and installs it if necessary.


It also installs:
- The JRE from [Eclipse Temurin](https://adoptium.net/fr/temurin/releases/)
- The [sample datasets](https://github.com/MLClustersML/mlclusters-samples/releases/latest).
- Documentation files:
  - PDF Guides .

## How to obtain the package assets
All the package assets (installers, documentation, etc) are available at the
[`mlclusters-win-install-assets`](https://github.com/MLClustersML/mlclusters-win-install-assets/releases/latest)
repository.

## How to build the installer manually
1) Install NSIS and make sure `makensis` it is available in the `%PATH%`.
2) Position the local repo to the desired tag (ex: `git checkout 10.2.0-rc1`)
3) Download and decompress the package assets to your machine, on `packaging\windows\nsis\assets`
4) [Build MLClusters in Release mode](https://github.com/MLClustersML/mlclusters/wiki/Building-MLClusters), to build the binaries and the jars
5) In a console, go to the `packaging/windows/nsis` directory and execute
```bat
%REM We assume the package assets were downloaded to packaging\windows\nsis\assets
%REM Before building the installer, update the following script arguments, mainly:
%REM - MLCLUSTERS_VERSION: MLClusters version for the installer.
%REM                   Should be identical to the value that is set in MLCLUSTERS_STR in src/Learning/KWUtils/KWMLClustersVersion.h
%REM - MLCLUSTERS_REDUCED_VERSION: MLClusters version without suffix and only digits and periods.
%REM                           Thus, the pre-release fields of MLCLUSTERS_VERSION are ignored in MLCLUSTERS_REDUCED_VERSION.

makensis ^
   /DMLCLUSTERS_VERSION=10.7.0-b.0 ^
   /DMLCLUSTERS_REDUCED_VERSION=10.7.0 ^
   /DMLCLUSTERS_WINDOWS_BUILD_DIR=..\..\..\build\Release ^
   /DJRE_PATH=.\assets\jre\ ^
   /DMSMPI_INSTALLER_PATH=.\assets\msmpisetup.exe ^
   /DMSMPI_VERSION=10.1.3 ^
   /DMLCLUSTERS_SAMPLES_DIR=.\assets\samples ^
   /DMLCLUSTERS_DOC_DIR=.\assets\doc ^
   mlclusters.nsi
```
The resulting installer will be at `packaging/windows/nsis/mlclusters-10.7.0-b.0-setup.exe`.

## Signature of binaries and installer
For a release version of the installer, the binaries and the installer need to be signed
4.bis) Sign the binary: `mlclusters.exe`
5.bis) Sign the installer


_Note 1_: See [below](#build-script-arguments) for the details of the installer builder script arguments.

_Note 2_: If your are using powershell replace the `^` characters by backticks `` ` `` in the
multi-line command.


## Github Workflow
This process is automatized in the [pack-nsis.yml workflow](../../../.github/workflows/pack-nsis.yml).

## Build script arguments
All the arguments are mandatory except for `DEBUG`, they must be prefixed by `/D` and post fixed by
`=<value>` to specify a value.

- `DEBUG`: Enables debug messages in the installer. They are "OK" message boxes.
- `MLCLUSTERS_VERSION`: MLClusters version for the installer.
- `MLCLUSTERS_REDUCED_VERSION`: MLClusters version without suffix and only digits and periods.
- `MLCLUSTERS_WINDOWS_BUILD_DIR`: Build directory for (usually `build` relative to
  the project root).
- `JRE_PATH`: Path to the Java Runtime Environment (JRE) directory.
- `MSMPI_INSTALLER_PATH`: Path to the Microsoft MPI (MS-MPI) installer.
- `MSMPI_MPI_VERSION`: MS-MPI version.
- `MLCLUSTERS_SAMPLES_DIR`: Path to the sample datasets directory.
- `MLCLUSTERS_DOC_DIR`: Path to the directory containing the documentation.
