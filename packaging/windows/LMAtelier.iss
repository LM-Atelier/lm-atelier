#ifndef MyAppVersion
  #error MyAppVersion must be supplied by the release build
#endif
#ifndef MyFileVersion
  #error MyFileVersion must be supplied by the release build
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
RedirectionGuard=yes
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
VersionInfoVersion={#MyFileVersion}
VersionInfoTextVersion={#MyAppVersion}
VersionInfoDescription=LM Atelier installer
VersionInfoProductName=LM Atelier
VersionInfoProductVersion={#MyFileVersion}
VersionInfoProductTextVersion={#MyAppVersion}
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

[Code]
var
  PurgeDataCheckBox: TNewCheckBox;

function UninstallParameterPresent(const Name: String): Boolean;
var
  Index: Integer;
begin
  Result := False;
  for Index := 1 to ParamCount do
  begin
    if CompareText(ParamStr(Index), Name) = 0 then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

procedure InitializeUninstallProgressForm;
begin
  PurgeDataCheckBox := TNewCheckBox.Create(UninstallProgressForm);
  PurgeDataCheckBox.Parent := UninstallProgressForm;
  PurgeDataCheckBox.Left := UninstallProgressForm.StatusLabel.Left;
  PurgeDataCheckBox.Top :=
    UninstallProgressForm.StatusLabel.Top + UninstallProgressForm.StatusLabel.Height + 12;
  PurgeDataCheckBox.Width := UninstallProgressForm.StatusLabel.Width;
  PurgeDataCheckBox.Caption := 'Delete chats, media, models, settings, and other local data';
  PurgeDataCheckBox.Checked := UninstallParameterPresent('/PURGEDATA');
end;

procedure CurUninstallStepChanged(CurrentStep: TUninstallStep);
var
  DataDirectory: String;
begin
  if (CurrentStep = usUninstall) and PurgeDataCheckBox.Checked then
  begin
    DataDirectory := ExpandConstant('{localappdata}\LMAtelier\data');
    if not DelTree(DataDirectory, True, True, True) then
      MsgBox(
        'LM Atelier could not remove all local data from ' + DataDirectory + '.',
        mbError,
        MB_OK
      );
  end;
end;
