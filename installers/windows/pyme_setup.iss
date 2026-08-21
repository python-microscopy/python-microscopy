; PYME Windows Installer — produced by InnoSetup on windows-latest.
;
; Typical CI invocation (after running install.bat to create the bundle):
;   set VERSION=<output of: python -c "import importlib.metadata; print(importlib.metadata.version('python-microscopy'))">
;   ISCC.exe /DAppSourceDir=C:\path\to\bundle /DAppVersion=%VERSION% pyme_setup.iss

#ifndef AppSourceDir
  #define AppSourceDir "..\..\PYME_bundle"
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName      "PYME"
#define AppPublisher "Baddeley Lab, University of Auckland"
; Icons ship with the package inside the venv tree.
; pymeLogo.png has no .ico equivalent — pmanal.ico is used in its place.
#define IconsDir     "{app}\venv\Lib\site-packages\PYME\resources\icons"

[Setup]
; AppId uniquely identifies this application for upgrades and uninstall — do not change.
AppId={{8F3A2E1D-6B4C-4D9F-A7E2-3C1B5F8A2D6E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; Per-user install by default; elevation dialog allows all-users install.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=PYME-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Avoid prompting to close running apps — PYME processes are independent.
CloseApplications=no

[Files]
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\PYMEAcquire";      Filename: "{app}\PYMEAcquire.cmd";      IconFilename: "{#IconsDir}\pmacquire.ico"
Name: "{group}\PYMEImage";        Filename: "{app}\PYMEImage.cmd";        IconFilename: "{#IconsDir}\pmanal.ico"
Name: "{group}\PYMEVis";          Filename: "{app}\PYMEVis.cmd";          IconFilename: "{#IconsDir}\pmvis.ico"
Name: "{group}\PYMEClusterOfOne"; Filename: "{app}\PYMEClusterOfOne.cmd"; IconFilename: "{#IconsDir}\pmanal.ico"
Name: "{group}\PYME Console";     Filename: "{app}\pyme-console.cmd";     IconFilename: "{#IconsDir}\pmanal.ico"
Name: "{group}\Uninstall PYME";   Filename: "{uninstallexe}"
