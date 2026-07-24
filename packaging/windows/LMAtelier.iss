#ifndef MyAppVersion
  #define MyAppVersion "0.1.3"
#endif
#ifndef MySourceDir
  #define MySourceDir "..\..\build\pyinstaller-windows\LM Atelier"
#endif
#ifndef MyOutputDir
  #define MyOutputDir "..\..\release"
#endif

#define MyAppName "LM Atelier"
#define MyAppPublisher "LM Atelier"
#define MyAppExeName "LM Atelier.exe"

[Setup]
AppId={{BC809AEF-330E-4CD3-820A-DA9E149A8DC2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\LM Atelier
DefaultGroupName=LM Atelier
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#MyOutputDir}
OutputBaseFilename=LM-Atelier-Setup-{#MyAppVersion}-windows-x86_64
SetupIconFile=..\..\build\installer-assets\lm-atelier.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
VersionInfoVersion={#MyAppVersion}
VersionInfoDescription=LM Atelier installer
VersionInfoProductName=LM Atelier
VersionInfoProductVersion={#MyAppVersion}
VersionInfoCompany=LM Atelier
LicenseFile=..\..\LICENSE

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LM Atelier"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\LM Atelier"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Open LM Atelier"; Flags: nowait postinstall skipifsilent
