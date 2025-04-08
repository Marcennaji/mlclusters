# MLClusters installer builder NSIS script

# Set Unicode to avoid warning 7998: "ANSI targets are deprecated"
Unicode True

# Set compresion to LZMA (faster)
SetCompressor /SOLID lzma
#SetCompress off

# Include NSIS librairies
!include "LogicLib.nsh"
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "x64.nsh"
!include "winmessages.nsh"

# Include Custom libraries
!include "MLClustersGlobals.nsh"
!include "MLClustersPrerequisiteFunc.nsh"
!include "ReplaceInFile.nsh"



# Definitions for registry change notification
!define SHCNE_ASSOCCHANGED 0x8000000
!define SHCNF_IDLIST 0

# Get installation folder from registry if available
InstallDirRegKey HKLM Software\mlclusters ""

# Request admin privileges
RequestExecutionLevel admin

# Make it aware of HiDPI screens
ManifestDPIAware true

# Macro to check input parameter definitions
!macro CheckInputParameter ParameterName
  !ifndef ${ParameterName}
    !error "${ParameterName} is not defined. Use the flag '-D${ParameterName}=...' to define it."
  !endif
!macroend

# Check the mandatory input definitions
!insertmacro CheckInputParameter MLCLUSTERS_VERSION
!insertmacro CheckInputParameter MLCLUSTERS_REDUCED_VERSION
!insertmacro CheckInputParameter MLCLUSTERS_WINDOWS_BUILD_DIR
!insertmacro CheckInputParameter JRE_PATH
!insertmacro CheckInputParameter MSMPI_INSTALLER_PATH
!insertmacro CheckInputParameter MSMPI_VERSION
!insertmacro CheckInputParameter MLCLUSTERS_SAMPLES_DIR
!insertmacro CheckInputParameter MLCLUSTERS_DOC_DIR

# Application name and installer file name
Name "MLClusters ${MLCLUSTERS_VERSION}"
OutFile "mlclusters-${MLCLUSTERS_VERSION}-setup.exe"

########################
# Variable definitions #
########################

# Requirements installation flags
Var /GLOBAL MPIInstallationNeeded

# Requirements installation messages
Var /GLOBAL MPIInstallationMessage

# Previous Uninstaller data
Var /GLOBAL PreviousUninstaller
Var /GLOBAL PreviousVersion

# %Public%, %AllUsersProfile% (%ProgramData%) and samples directory
Var /GLOBAL WinPublicDir
Var /GLOBAL AllUsersProfileDir
Var /GLOBAL GlobalMLClustersDataDir
Var /GLOBAL SamplesInstallDir

# Root key for the uninstaller in the windows registry
!define UninstallerKey "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"

#####################################
# Modern UI Interface Configuration #
#####################################

# General configuration
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP ".\images\headerimage.bmp"
!define MUI_HEADERIMAGE_LEFT
!define MUI_WELCOMEFINISHPAGE_BITMAP ".\images\welcomefinish.bmp"
!define MUI_ABORTWARNING
!define MUI_ICON ".\images\installer.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\win-uninstall.ico"
BrandingText "MLClusters ${MLCLUSTERS_VERSION}"

# Welcome page
!define MUI_WELCOMEPAGE_TITLE "Welcome to the MLClusters ${MLCLUSTERS_VERSION} Setup Wizard"
!define MUI_WELCOMEPAGE_TEXT \
    "MLClusters is a machine learning clustering tool based on the Khiops ML libraries.$\r$\n$\r$\n$\r$\n$\r$\n$(MUI_${MUI_PAGE_UNINSTALLER_PREFIX}TEXT_WELCOME_INFO_TEXT)"
!insertmacro MUI_PAGE_WELCOME

# Licence page
!insertmacro MUI_PAGE_LICENSE "..\..\..\LICENSE"

# Custom page for requirements software
Page custom RequirementsPageShow RequirementsPageLeave

# Install directory choice page
!insertmacro MUI_PAGE_DIRECTORY

# Install files choice page
!insertmacro MUI_PAGE_INSTFILES

# Final page
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Create desktop shortcut"
!define MUI_FINISHPAGE_RUN_FUNCTION "CreateDesktopShortcuts"
!define MUI_FINISHPAGE_TEXT "$\r$\n$\r$\nThank you for installing MLClusters."
!define MUI_FINISHPAGE_LINK "mlclusters.org"
!define MUI_FINISHPAGE_LINK_LOCATION "https://mlclusters.org"
!insertmacro MUI_PAGE_FINISH

# Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

# Language (must be defined after uninstaller)
!insertmacro MUI_LANGUAGE "English"

#######################
# Version Information #
#######################

VIProductVersion "${MLCLUSTERS_REDUCED_VERSION}.0"
VIAddVersionKey /LANG=${LANG_ENGLISH} "ProductName" "MLClusters"
VIAddVersionKey /LANG=${LANG_ENGLISH} "CompanyName" "Open Source Community"
VIAddVersionKey /LANG=${LANG_ENGLISH} "LegalCopyright" "Copyright © 2025 Open Source Contributors"
VIAddVersionKey /LANG=${LANG_ENGLISH} "FileDescription" "MLClusters Installer"
VIAddVersionKey /LANG=${LANG_ENGLISH} "FileVersion" "${MLCLUSTERS_VERSION}"

######################
# Installer Sections #
######################

Section "Install" SecInstall
  # In order to have shortcuts and documents for all users
  SetShellVarContext all
  
  # Detect Java
  Call RequirementsDetection


  # MPI installation is always required, because MLClusters is linked with MPI DLL
  ${If} $MPIInstallationNeeded == "1"
      Call InstallMPI
  ${EndIf}

  # Activate file overwrite
  SetOverwrite on

  # Install executables and java libraries
  SetOutPath "$INSTDIR\bin"
  File "${MLCLUSTERS_WINDOWS_BUILD_DIR}\Release\mlclusters.exe"
   File "${MLCLUSTERS_WINDOWS_BUILD_DIR}\bin\_khiopsgetprocnumber.exe"
  File "${MLCLUSTERS_WINDOWS_BUILD_DIR}\jars\norm.jar"
  File "${MLCLUSTERS_WINDOWS_BUILD_DIR}\jars\khiops.jar"
  File "..\mlclusters_env.cmd"
  File "..\mlclusters.cmd"
 
  # Install Docs
  SetOutPath "$INSTDIR"
  File "/oname=LICENSE.txt" "..\..\..\LICENSE"
  SetOutPath "$INSTDIR\doc"
  File /nonfatal /a /r "${MLCLUSTERS_DOC_DIR}\"

  # Install icons
  SetOutPath "$INSTDIR\bin\icons"
  File ".\images\installer.ico"
 
  # Set the samples directory to be located either within %PUBLIC% or %ALLUSERSPROFILE% as fallback
  ReadEnvStr $WinPublicDir PUBLIC
  ReadEnvStr $AllUsersProfileDir ALLUSERSPROFILE
  ${If} $WinPublicDir != ""
    StrCpy $GlobalMLClustersDataDir "$WinPublicDir\mlclusters_data"
  ${ElseIf} $AllUsersProfileDir != ""
    StrCpy $GlobalMLClustersDataDir "$AllUsersProfileDir\mlclusters_data"
  ${Else}
    StrCpy $GlobalMLClustersDataDir ""
  ${EndIf}

  # Debug message
  !ifdef DEBUG
    ${If} $GlobalMLClustersDataDir == ""
      Messagebox MB_OK "Could find PUBLIC nor ALLUSERSPROFILE directories. Samples not installed."
    ${Else}
      Messagebox MB_OK "Samples will be installed at $GlobalMLClustersDataDir\samples."
    ${EndIf}
  !endif

  # Install samples only if the directory is defined
  ${If} $GlobalMLClustersDataDir != ""
    StrCpy $SamplesInstallDir "$GlobalMLClustersDataDir\samples"
    SetOutPath "$SamplesInstallDir"
    File "/oname=README.txt" "${MLCLUSTERS_SAMPLES_DIR}\README.md"
    SetOutPath "$SamplesInstallDir\Adult"
    File "${MLCLUSTERS_SAMPLES_DIR}\Adult\Adult.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\Adult\Adult.txt"
    SetOutPath "$SamplesInstallDir\Iris"
    File "${MLCLUSTERS_SAMPLES_DIR}\Iris\Iris.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\Iris\Iris.txt"
    SetOutPath "$SamplesInstallDir\Mushroom"
    File "${MLCLUSTERS_SAMPLES_DIR}\Mushroom\Mushroom.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\Mushroom\Mushroom.txt"
    SetOutPath "$SamplesInstallDir\Letter"
    File "${MLCLUSTERS_SAMPLES_DIR}\Letter\Letter.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\Letter\Letter.txt"
    SetOutPath "$SamplesInstallDir\SpliceJunction"
    File "${MLCLUSTERS_SAMPLES_DIR}\SpliceJunction\SpliceJunction.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\SpliceJunction\SpliceJunction.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\SpliceJunction\SpliceJunctionDNA.txt"
    SetOutPath "$SamplesInstallDir\Accidents"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\Accidents.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\Accidents.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\Places.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\Users.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\Vehicles.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\train.py"
    File "/oname=README.txt" "${MLCLUSTERS_SAMPLES_DIR}\Accidents\README.md"
    SetOutPath "$SamplesInstallDir\Accidents\raw"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\AccidentsPreprocess.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\Description_BD_ONISR.pdf"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\Licence_Ouverte.pdf"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\caracteristiques-2018.csv"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\lieux-2018.csv"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\usagers-2018.csv"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\vehicules-2018.csv"
    File "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\preprocess.py"
    File "/oname=README.txt" "${MLCLUSTERS_SAMPLES_DIR}\Accidents\raw\README.md"
    SetOutPath "$SamplesInstallDir\AccidentsSummary"
    File "${MLCLUSTERS_SAMPLES_DIR}\AccidentsSummary\Accidents.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\AccidentsSummary\Accidents.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\AccidentsSummary\Vehicles.txt"
    File "/oname=README.txt" "${MLCLUSTERS_SAMPLES_DIR}\AccidentsSummary\README.md"
    SetOutPath "$SamplesInstallDir\Customer"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\Customer.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\CustomerRecoded.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\Customer.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\Address.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\Service.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\Usage.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\sort_and_recode_customer.py"
    File "/oname=README.txt" "${MLCLUSTERS_SAMPLES_DIR}\Customer\README.md"
    SetOutPath "$SamplesInstallDir\Customer\unsorted"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\unsorted\Customer-unsorted.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\unsorted\Address-unsorted.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\unsorted\Service-unsorted.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\Customer\unsorted\Usage-unsorted.txt"
    SetOutPath "$SamplesInstallDir\CustomerExtended"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\Customer.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\CustomerRecoded.kdic"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\Customer.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\Address.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\Service.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\Usage.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\City.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\Country.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\Product.txt"
    File "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\recode_customer.py"
    File "/oname=README.txt" "${MLCLUSTERS_SAMPLES_DIR}\CustomerExtended\README.md"
  ${EndIf}

  # Install JRE
  SetOutPath $INSTDIR\jre
  File /nonfatal /a /r "${JRE_PATH}\"

  
  # Add the installer file
  SetOutPath $TEMP
  
  
  #############################
  # Finalize the installation #
  #############################

  # Setting up the GUI in mlclusters_env.cmd: replace @GUI_STATUS@ by "true" in the installed file
  Push @GUI_STATUS@ 
  Push 'true' 
  Push all 
  Push all 
  Push $INSTDIR\bin\mlclusters_env.cmd
  Call ReplaceInFile

  # Setting up MPI in mlclusters_env.cmd: replace @SET_MPI@ by "SET_MPI_SYSTEM_WIDE" in the installed file
  Push @SET_MPI@
  Push SET_MPI_SYSTEM_WIDE 
  Push all 
  Push all 
  Push $INSTDIR\bin\mlclusters_env.cmd
  Call ReplaceInFile

  # Setting up IS_CONDA_VAR variable in mlclusters_env.cmd: replace @SET_MPI@ by an empty string: this is not an installer for conda
  Push @IS_CONDA_VAR@
  Push "" 
  Push all 
  Push all 
  Push $INSTDIR\bin\mlclusters_env.cmd
  Call ReplaceInFile

  # Create the MLClusters shell
  FileOpen $0 "$INSTDIR\bin\shell_mlclusters.cmd" w
  FileWrite $0 '@echo off$\r$\n'
  FileWrite $0 'REM Open a shell session with access to MLClusters$\r$\n'
  FileWrite $0 `if "%MLCLUSTERS_HOME%".=="". set MLCLUSTERS_HOME=$INSTDIR$\r$\n`
  FileWrite $0 'set path=%MLCLUSTERS_HOME%\bin;%path%$\r$\n'
  FileWrite $0 'title Shell MLClusters$\r$\n'
  FileWrite $0 '%comspec% /K "echo Welcome to MLClusters scripting mode & echo Type mlclusters -h to get help'
  FileClose $0

  # Create the uninstaller
  WriteUninstaller "$INSTDIR\uninstall-mlclusters.exe"


  #####################################
  # Windows environment customization #
  # ###################################

  # Write registry keys to add MLClusters in the Add/Remove Programs pane
  WriteRegStr HKLM "Software\MLClusters" "" $INSTDIR
  WriteRegStr HKLM "${UninstallerKey}\MLClusters" "UninstallString" '"$INSTDIR\uninstall-mlclusters.exe"'
  WriteRegStr HKLM "${UninstallerKey}\MLClusters" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${UninstallerKey}\MLClusters" "DisplayName" "MLClusters"
  WriteRegStr HKLM "${UninstallerKey}\MLClusters" "Publisher" "Open Source Community"
  WriteRegStr HKLM "${UninstallerKey}\MLClusters" "DisplayIcon" "$INSTDIR\bin\icons\installer.ico"
  WriteRegStr HKLM "${UninstallerKey}\MLClusters" "DisplayVersion" "${MLCLUSTERS_VERSION}"
  WriteRegStr HKLM "${UninstallerKey}\MLClusters" "URLInfoAbout" "https://github.com/Marcennaji/mlclusters"
  WriteRegDWORD HKLM "${UninstallerKey}\MLClusters" "NoModify" "1"
  WriteRegDWORD HKLM "${UninstallerKey}\MLClusters" "NoRepair" "1"

  # Set as the startup dir for all executable shortcuts (yes it is done with SetOutPath!)
  ${If} $GlobalMLClustersDataDir != ""
    SetOutPath $GlobalMLClustersDataDir
  ${Else}
    SetOutPath $INSTDIR
  ${EndIf}

  # Create application shortcuts in the installation directory
  DetailPrint "Installing Start menu Shortcut..."
  CreateShortCut "$INSTDIR\MLClusters.lnk" "$INSTDIR\bin\mlclusters.cmd" "" "$INSTDIR\bin\icons\mlclusters.ico" 0 SW_SHOWMINIMIZED
  ExpandEnvStrings $R0 "%COMSPEC%"
  CreateShortCut "$INSTDIR\Shell MLClusters.lnk" "$INSTDIR\bin\shell_mlclusters.cmd" "" "$R0"

  # Create start menu shortcuts for the executables and documentation
  DetailPrint "Installing Start menu Shortcut..."
  CreateDirectory "$SMPROGRAMS\MLClusters"
  CreateShortCut "$SMPROGRAMS\MLClusters\MLClusters.lnk" "$INSTDIR\bin\mlclusters.cmd" "" "$INSTDIR\bin\icons\mlclusters.ico" 0 SW_SHOWMINIMIZED
  ExpandEnvStrings $R0 "%COMSPEC%"
  CreateShortCut "$SMPROGRAMS\MLClusters\Shell MLClusters.lnk" "$INSTDIR\bin\shell_mlclusters.cmd" "" "$R0"
  CreateShortCut "$SMPROGRAMS\MLClusters\Uninstall.lnk" "$INSTDIR\uninstall-mlclusters.exe"
  CreateDirectory "$SMPROGRAMS\MLClusters\doc"
  CreateShortCut "$SMPROGRAMS\MLClusters\doc\Tutorial.lnk" "$INSTDIR\doc\MLClustersTutorial.pdf"
  CreateShortCut "$SMPROGRAMS\MLClusters\doc\MLClusters.lnk" "$INSTDIR\doc\MLClustersGuide.pdf"
  SetOutPath "$INSTDIR"

  # Define aliases for the following registry keys (also used in the uninstaller section)
  # - HKLM (all users)
  # - HKCU (current user)
  !define env_hklm 'HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"'
  !define env_hkcu 'HKCU "Environment"'

  # Set MLCLUSTERS_HOME for the local machine and current user
  WriteRegExpandStr ${env_hklm} "MLCLUSTERS_HOME" "$INSTDIR"
  WriteRegExpandStr ${env_hkcu} "MLCLUSTERS_HOME" "$INSTDIR"

  # Make sure windows knows about the change
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  # MLClusters dictionary file extension
  ReadRegStr $R0 HKCR ".kdic" ""
  ${if} $R0 == "MLClusters.Dictionary.File"
    DeleteRegKey HKCR "MLClusters.Dictionary.File"
  ${EndIf}
  WriteRegStr HKCR ".kdic" "" "MLClusters.Dictionary.File"
  WriteRegStr HKCR "MLClusters.Dictionary.File" "" "MLClusters Dictionary File"
  ReadRegStr $R0 HKCR "MLClusters.Dictionary.File\shell\open\command" ""
  ${If} $R0 == ""
    WriteRegStr HKCR "MLClusters.Dictionary.File\shell" "" "open"
    WriteRegStr HKCR "MLClusters.Dictionary.File\shell\open\command" "" 'notepad.exe "%1"'
  ${EndIf}

  # MLClusters scenario file
  ReadRegStr $R0 HKCR "._kh" ""
  ${if} $R0 == "MLClusters.File"
    DeleteRegKey HKCR "MLClusters.File"
  ${EndIf}
  WriteRegStr HKCR "._kh" "" "MLClusters.File"
  WriteRegStr HKCR "MLClusters.File" "" "MLClusters File"
  WriteRegStr HKCR "MLClusters.File\DefaultIcon" "" "$INSTDIR\bin\icons\mlclusters.ico"
  ReadRegStr $R0 HKCR "MLClusters.File\shell\open\command" ""
  ${If} $R0 == ""
    WriteRegStr HKCR "MLClusters.File\shell" "" "open"
    WriteRegStr HKCR "MLClusters.File\shell\open\command" "" 'notepad.exe "%1"'
  ${EndIf}
  WriteRegStr HKCR "MLClusters.File\shell\compile" "" "Execute MLClusters Script"
  WriteRegStr HKCR "MLClusters.File\shell\compile\command" "" '"$INSTDIR\bin\mlclusters.cmd" -i "%1"'

    # Notify the file extension changes
  System::Call 'Shell32::SHChangeNotify(i ${SHCNE_ASSOCCHANGED}, i ${SHCNF_IDLIST}, i 0, i 0)'

  # Debug message
  !ifdef DEBUG
    Messagebox MB_OK "Installation finished!"
  !endif

SectionEnd


###############
# Uninstaller #
###############

Section "Uninstall"
  # In order to have shortcuts and documents for all users
  SetShellVarContext all

  # Restore Registry #
  # Unregister file associations
  DetailPrint "Uninstalling MLClusters Shell Extensions..."

  # Unregister MLClusters dictionary file extension
  ${If} $R0 == "MLClusters.Dictionary.File"
    DeleteRegKey HKCR ".kdic"
  ${EndIf}
  DeleteRegKey HKCR "MLClusters.Dictionary.File"

  # Unregister MLClusters file extension
  ${If} $R0 == "MLClusters.File"
    DeleteRegKey HKCR "._kh"
  ${EndIf}
  DeleteRegKey HKCR "MLClusters.File"

  # Notify file extension changes
  System::Call 'Shell32::SHChangeNotify(i ${SHCNE_ASSOCCHANGED}, i ${SHCNF_IDLIST}, i 0, i 0)'

  # Delete installation folder key
  DeleteRegKey HKLM "${UninstallerKey}\MLClusters"
  DeleteRegKey HKLM "Software\MLClusters"

  # Delete environement variable MLCLUSTERS_HOME
  DeleteRegValue ${env_hklm} "MLCLUSTERS_HOME"
  DeleteRegValue ${env_hkcu} "MLCLUSTERS_HOME"

  # Delete deprecated environment variable MLClustersHome
  DeleteRegValue ${env_hklm} "MLClustersHome"
  DeleteRegValue ${env_hkcu} "MLClustersHome"

  # Make sure windows knows about the changes in the environment
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  # Delete files #
  # Note: Some directories are removed only if they are completely empty (no "/r" RMDir flag)
  DetailPrint "Deleting Files ..."

  # Delete docs
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\WHATSNEW.txt"
  RMDir /r "$INSTDIR\doc"

  # Delete jre
  RMDir /r "$INSTDIR\jre"

  # Delete icons
  RMDir /r "$INSTDIR\bin\icons"

  # Delete executables and scripts
  Delete "$INSTDIR\bin\mlclusters_env.cmd"
  Delete "$INSTDIR\bin\mlclusters.cmd"
  Delete "$INSTDIR\bin\mlclusters.exe"
  Delete "$INSTDIR\bin\_khiopsgetprocnumber.exe"
  Delete "$INSTDIR\bin\norm.jar"
  Delete "$INSTDIR\bin\khiops.jar"
  Delete "$INSTDIR\bin\shell_mlclusters.cmd"
  RMDir "$INSTDIR\bin"

  # Delete shortcuts from install dir
  Delete "$INSTDIR\MLClusters.lnk"
  Delete "$INSTDIR\Shell MLClusters.lnk"

  # Delete the installer
  Delete "$INSTDIR\uninstall-mlclusters.exe"

  # Remove install directory
  RMDir "$INSTDIR"

  # Delete desktop shortcuts
  Delete "$DESKTOP\MLClusters.lnk"
  Delete "$DESKTOP\Shell MLClusters.lnk"

  # Delete Start Menu Shortcuts
  RMDir /r "$SMPROGRAMS\MLClusters"

  # Set the samples directory to be located either within %PUBLIC% or %ALLUSERSPROFILE% as fallback
  ReadEnvStr $WinPublicDir PUBLIC
  ReadEnvStr $AllUsersProfileDir ALLUSERSPROFILE
  ${If} $WinPublicDir != ""
    StrCpy $GlobalMLClustersDataDir "$WinPublicDir\mlclusters_data"
  ${ElseIf} $AllUsersProfileDir != ""
    StrCpy $GlobalMLClustersDataDir "$AllUsersProfileDir\mlclusters_data"
  ${Else}
    StrCpy $GlobalMLClustersDataDir ""
  ${EndIf}

  # Delete sample datasets
  # We do not remove the whole directory to save the users results from MLClusters' analyses
  ${If} $GlobalMLClustersDataDir != ""
    StrCpy $SamplesInstallDir "$GlobalMLClustersDataDir\samples"
    Delete "$SamplesInstallDir\AccidentsSummary\Accidents.kdic"
    Delete "$SamplesInstallDir\AccidentsSummary\Accidents.txt"
    Delete "$SamplesInstallDir\AccidentsSummary\README.txt"
    Delete "$SamplesInstallDir\AccidentsSummary\Vehicles.txt"
    Delete "$SamplesInstallDir\Accidents\Accidents.kdic"
    Delete "$SamplesInstallDir\Accidents\Accidents.txt"
    Delete "$SamplesInstallDir\Accidents\Places.txt"
    Delete "$SamplesInstallDir\Accidents\README.txt"
    Delete "$SamplesInstallDir\Accidents\Users.txt"
    Delete "$SamplesInstallDir\Accidents\Vehicles.txt"
    Delete "$SamplesInstallDir\Accidents\raw\AccidentsPreprocess.kdic"
    Delete "$SamplesInstallDir\Accidents\raw\Description_BD_ONISR.pdf"
    Delete "$SamplesInstallDir\Accidents\raw\Licence_Ouverte.pdf"
    Delete "$SamplesInstallDir\Accidents\raw\README.txt"
    Delete "$SamplesInstallDir\Accidents\raw\caracteristiques-2018.csv"
    Delete "$SamplesInstallDir\Accidents\raw\lieux-2018.csv"
    Delete "$SamplesInstallDir\Accidents\raw\preprocess.py"
    Delete "$SamplesInstallDir\Accidents\raw\usagers-2018.csv"
    Delete "$SamplesInstallDir\Accidents\raw\vehicules-2018.csv"
    Delete "$SamplesInstallDir\Accidents\train.py"
    Delete "$SamplesInstallDir\Adult\Adult.kdic"
    Delete "$SamplesInstallDir\Adult\Adult.txt"
    Delete "$SamplesInstallDir\CustomerExtended\Address.txt"
    Delete "$SamplesInstallDir\CustomerExtended\City.txt"
    Delete "$SamplesInstallDir\CustomerExtended\Country.txt"
    Delete "$SamplesInstallDir\CustomerExtended\Customer.kdic"
    Delete "$SamplesInstallDir\CustomerExtended\Customer.txt"
    Delete "$SamplesInstallDir\CustomerExtended\CustomerRecoded.kdic"
    Delete "$SamplesInstallDir\CustomerExtended\Product.txt"
    Delete "$SamplesInstallDir\CustomerExtended\README.txt"
    Delete "$SamplesInstallDir\CustomerExtended\Service.txt"
    Delete "$SamplesInstallDir\CustomerExtended\Usage.txt"
    Delete "$SamplesInstallDir\CustomerExtended\recode_customer.py"
    Delete "$SamplesInstallDir\Customer\Address.txt"
    Delete "$SamplesInstallDir\Customer\Customer.kdic"
    Delete "$SamplesInstallDir\Customer\Customer.txt"
    Delete "$SamplesInstallDir\Customer\CustomerRecoded.kdic"
    Delete "$SamplesInstallDir\Customer\README.txt"
    Delete "$SamplesInstallDir\Customer\Service.txt"
    Delete "$SamplesInstallDir\Customer\Usage.txt"
    Delete "$SamplesInstallDir\Customer\sort_and_recode_customer.py"
    Delete "$SamplesInstallDir\Customer\unsorted\Address-unsorted.txt"
    Delete "$SamplesInstallDir\Customer\unsorted\Customer-unsorted.txt"
    Delete "$SamplesInstallDir\Customer\unsorted\Service-unsorted.txt"
    Delete "$SamplesInstallDir\Customer\unsorted\Usage-unsorted.txt"
    Delete "$SamplesInstallDir\Iris\Iris.kdic"
    Delete "$SamplesInstallDir\Iris\Iris.txt"
    Delete "$SamplesInstallDir\Letter\Letter.kdic"
    Delete "$SamplesInstallDir\Letter\Letter.txt"
    Delete "$SamplesInstallDir\Mushroom\Mushroom.kdic"
    Delete "$SamplesInstallDir\Mushroom\Mushroom.txt"
    Delete "$SamplesInstallDir\README.txt"
    Delete "$SamplesInstallDir\SpliceJunction\SpliceJunction.kdic"
    Delete "$SamplesInstallDir\SpliceJunction\SpliceJunction.txt"
    Delete "$SamplesInstallDir\SpliceJunction\SpliceJunctionDNA.txt"
    RMDir "$SamplesInstallDir\AccidentsSummary\"
    RMDir "$SamplesInstallDir\Accidents\raw\"
    RMDir "$SamplesInstallDir\Accidents\"
    RMDir "$SamplesInstallDir\Adult\"
    RMDir "$SamplesInstallDir\CustomerExtended\"
    RMDir "$SamplesInstallDir\Customer\unsorted\"
    RMDir "$SamplesInstallDir\Customer\"
    RMDir "$SamplesInstallDir\Iris\"
    RMDir "$SamplesInstallDir\Letter\"
    RMDir "$SamplesInstallDir\Mushroom\"
    RMDir "$SamplesInstallDir\SpliceJunction\"
    RMDir "$SamplesInstallDir"
  ${EndIf}
SectionEnd


#######################
# Installer Functions #
#######################

Function "CreateDesktopShortcuts"
  # Set as the startup dir for all executable shortcuts (yes it is done with SetOutPath!)
  ${If} $GlobalMLClustersDataDir != ""
    SetOutPath $GlobalMLClustersDataDir
  ${Else}
    SetOutPath $INSTDIR
  ${EndIf}

  # Create the shortcuts
  DetailPrint "Installing Desktop Shortcut..."
  CreateShortCut "$DESKTOP\MLClusters.lnk" "$INSTDIR\bin\mlclusters.cmd" "" "$INSTDIR\bin\icons\mlclusters.ico" 0 SW_SHOWMINIMIZED
FunctionEnd

# Predefined initialization install function
Function .onInit

  # Read location of the uninstaller
  ReadRegStr $PreviousUninstaller HKLM "${UninstallerKey}\MLClusters" "UninstallString"
  ReadRegStr $PreviousVersion HKLM "${UninstallerKey}\MLClusters" "DisplayVersion"

  # Ask the user to proceed if there was already a previous MLClusters version installed
  # In silent mode: remove previous version
  ${If} $PreviousUninstaller != ""
    MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION \
       "MLClusters $PreviousVersion is already installed. $\n$\nClick OK to remove the \
       previous version $\n$\nor Cancel to cancel this upgrade." \
       /SD IDOK IDOK uninst
    Abort

    # Run the uninstaller
    uninst:
    ClearErrors
    ExecWait '$PreviousUninstaller /S _?=$INSTDIR'

    # Run again the uninstaller to delete the uninstaller itself and the root dir (without waiting)
    # Must not be used in silent mode (may delete files from silent following installation)
    ${IfNot} ${Silent}
       ExecWait '$PreviousUninstaller /S'
    ${EndIf}
  ${EndIf}

  # Choice of default installation directory, for windows 32 or 64
  ${If} $INSTDIR == ""
    ${If} ${RunningX64}
      StrCpy $INSTDIR "$PROGRAMFILES64\mlclusters"
      # No 32-bit install
    ${EndIf}
  ${EndIf}
FunctionEnd


# Function to show the page for requirements
Function RequirementsPageShow
  # Detect requirements
  Call RequirementsDetection

  # Creation of page, with title and subtitle
  nsDialogs::Create 1018
  !insertmacro MUI_HEADER_TEXT "Check software requirements" "Check Microsoft MPI"

  # Message to show for the Microsoft MPI installation
  ${NSD_CreateLabel} 0 20u 100% 10u $MPIInstallationMessage

  # Show page
  nsDialogs::Show
FunctionEnd


# Requirements detection
# - Detects if the system architecture is 64-bit
# - Detects whether Java JRE and MPI are installed and their versions
Function RequirementsDetection
  # Abort installation if the machine does not have 64-bit architecture
  ${IfNot} ${RunningX64}
    Messagebox MB_OK "MLClusters works only on Windows 64 bits: installation will be terminated." /SD IDOK
    Quit
  ${EndIf}

  # Decide if MPI is required by detecting the number of cores
  # Note: This call defines MPIInstalledVersion
  Call DetectAndLoadMPIEnvironment

  # Try to install MPI
  StrCpy $MPIInstallationNeeded "0"
  StrCpy $MPIInstallationMessage ""
 
  # If it is not installed install it
  ${If} $MPIInstalledVersion == "0"
      StrCpy $MPIInstallationMessage "Microsoft MPI version ${MSMPI_VERSION} will be installed"
      StrCpy $MPIInstallationNeeded "1"
  # Otherwise install only if the required version is newer than the installed one
  ${Else}
      ${VersionCompare} "${MPIRequiredVersion}" "$MPIInstalledVersion" $0
      ${If} $0 == 1
        StrCpy $MPIInstallationMessage "Microsoft MPI will be upgraded to version ${MSMPI_VERSION}"
        StrCpy $MPIInstallationNeeded "1"
      ${Else}
        StrCpy $MPIInstallationMessage "Microsoft MPI version already installed"
      ${EndIf}
  ${EndIf}
 

  # Show debug information
  !ifdef DEBUG
    Messagebox MB_OK "MS-MPI: needed=$MPIInstallationNeeded required=${MPIRequiredVersion} installed=$MPIInstalledVersion"
  !endif

FunctionEnd

# No leave page for required software
Function RequirementsPageLeave
FunctionEnd
