import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Cpu, Folder, HardDrive, Plus } from "lucide-react";
import { useRef, useState } from "react";
import { api } from "./api";
import { AccessibleDialog } from "./AccessibleDialog";
import { CopyTextButton } from "./CopyTextButton";
import { SettingControl } from "./SettingControl";
import { CredentialSettingsCard } from "./CredentialSettingsCard";
import { DownloadDiagnosticsButton } from "./DownloadDiagnosticsButton";
import { ErrorCallout } from "./ErrorCallout";
import { RuntimeSetupCard } from "./RuntimeSetupCard";
import { StatusDot } from "./StatusDot";
import { WorkerLogFolderButton, WorkerStartupLimit } from "./WorkerStartupLimit";
import { WorkerStatusCard } from "./WorkerStatusCard";
import { downloadJson, formatBytes, formatDate, supportLinks } from "./format";
import {
  resolveCapabilitySettings,
  visibilityRank,
  type Visibility,
} from "./settings";
import { useConfirm } from "./useConfirm";
import type {
  ApplicationInfo,
  BackupInfo,
  EngineCapabilities,
  GenerationPreset,
  GenerationPresetBundle,
  ModelProfile,
  ModelProfileBundle,
  RuntimeStatus,
  SystemInfo,
} from "./types";

/** The settings view, with the editors and the formatter only it uses.
 *
 * Three hundred lines of surface the shell never needed to know about. It is
 * a view like the others and now sits with them.
 */

function formatTechnicalDetails(
  application: ApplicationInfo,
  system: SystemInfo,
  engines: EngineCapabilities[],
): string {
  const devices = system.devices.length
    ? [
        "Devices:",
        ...system.devices.map((device) => {
          const facts = [
            device.kind,
            device.backend,
            device.total_memory_bytes == null ? null : formatBytes(device.total_memory_bytes),
          ].filter(Boolean);
          return `- ${device.name}${facts.length ? ` (${facts.join("; ")})` : ""}`;
        }),
      ]
    : ["Devices: None detected"];
  const runtimes = engines.length
    ? [
        "Engines:",
        ...engines.map(
          (engine) => `- ${engine.engine} ${engine.version} (${engine.roles.join(", ")})`,
        ),
      ]
    : ["Engines: None reported"];
  return [
    `LM Atelier: ${application.version}`,
    `Platform: ${system.distribution} ${system.distribution_version} (${system.platform} ${system.platform_release}; ${system.architecture})`,
    `Runtime: Python ${system.python_version}`,
    `CPU: ${system.cpu_model}`,
    ...devices,
    ...runtimes,
  ].join("\n");
}

function ProfileEditor({
  profile,
  engines,
  onClose,
}: {
  profile: ModelProfile;
  engines: EngineCapabilities[];
  onClose: () => void;
}) {
  const client = useQueryClient();
  const [name, setName] = useState(profile.name);
  const [useCase, setUseCase] = useState(profile.use_case);
  const [isDefault, setIsDefault] = useState(profile.is_default);
  const [loadSettings, setLoadSettings] = useState(profile.load_settings_json);
  const [requestSettings, setRequestSettings] = useState(profile.request_settings_json);
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const refresh = () => void client.invalidateQueries({ queryKey: ["profiles"] });
  const save = useMutation({
    mutationFn: () => api.updateProfile(profile.id, {
      name,
      use_case: useCase,
      is_default: isDefault,
      load_settings: loadSettings,
      request_settings: requestSettings,
    }),
    onSuccess: () => { refresh(); onClose(); },
  });
  const clone = useMutation({
    mutationFn: () => api.cloneProfile(profile.id),
    onSuccess: () => { refresh(); onClose(); },
  });
  const reset = useMutation({
    mutationFn: () => api.resetProfile(profile.id),
    onSuccess: (value) => {
      setLoadSettings(value.load_settings_json);
      setRequestSettings(value.request_settings_json);
      refresh();
    },
  });
  const remove = useMutation({
    mutationFn: () => api.deleteProfile(profile.id),
    onSuccess: () => { refresh(); onClose(); },
  });
  const exportBundle = useMutation({
    mutationFn: () => api.exportProfile(profile.id),
    onSuccess: (bundle) => downloadJson(bundle, `${profile.name.replaceAll(/[^a-z0-9]+/gi, "-").toLowerCase()}.lm-atelier-profile.json`),
  });
  const engine = engines.find((item) => item.engine === profile.engine && item.roles.includes(profile.role))
    ?? engines.find((item) => item.roles.includes(profile.role));
  const fields = resolveCapabilitySettings(engine, profile.role).filter(
    (field) => visibilityRank[field.visibility] <= visibilityRank[visibility] && field.available,
  );
  const error = save.error ?? clone.error ?? reset.error ?? remove.error ?? exportBundle.error;
  return (
    <AccessibleDialog
      title="Edit profile"
      eyebrow={`${profile.role} profile · ${profile.engine}`}
      closeLabel="Close profile editor"
      onClose={onClose}
      className="settings-editor"
    >
      <label>Profile name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>Best used for<textarea rows={3} value={useCase} onChange={(event) => setUseCase(event.target.value)} placeholder="Programming, code review, technical explanations" /></label>
      <label className="toggle-row"><span><strong>Default {profile.role} model</strong><small>Used by chats set to Default. Auto uses it when no use case matches.</small></span><input type="checkbox" checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} /></label>
      <div className="segmented compact" role="group" aria-label="Profile setting detail">
        {(["basic", "advanced", "expert"] as Visibility[]).map((level) => <button type="button" key={level} className={visibility === level ? "active" : ""} aria-pressed={visibility === level} onClick={() => setVisibility(level)}>{level}</button>)}
      </div>
      <div className="settings-list embedded">
        {fields.map((field) => {
          const target = field.scope === "load" ? loadSettings : requestSettings;
          return <div className="scoped-setting" key={`${field.scope}:${field.key}:${JSON.stringify(target[field.key])}`}><span className="scope-label">{field.scope}{field.restart_required ? " · restart required" : ""}</span><SettingControl field={field} value={target[field.key] ?? field.default} onChange={(value) => field.scope === "load" ? setLoadSettings({ ...loadSettings, [field.key]: value }) : setRequestSettings({ ...requestSettings, [field.key]: value })} /></div>;
        })}
        {!engine && <p className="muted">No capability schema is available for this profile engine.</p>}
      </div>
      {error && <ErrorCallout message={error.message} />}
      <footer className="editor-actions"><button className="secondary danger" onClick={() => remove.mutate()} disabled={remove.isPending}>Delete profile</button><button className="secondary" onClick={() => reset.mutate()} disabled={reset.isPending}>Reset settings</button><button className="secondary" onClick={() => exportBundle.mutate()}>Export</button><button className="secondary" onClick={() => clone.mutate()}>Clone</button><button className="primary" onClick={() => save.mutate()} disabled={!name.trim() || save.isPending}>Save profile</button></footer>
    </AccessibleDialog>
  );
}

function PresetEditor({
  preset,
  engines,
  onClose,
}: {
  preset: GenerationPreset;
  engines: EngineCapabilities[];
  onClose: () => void;
}) {
  const client = useQueryClient();
  const [name, setName] = useState(preset.name);
  const [isDefault, setIsDefault] = useState(preset.is_default);
  const [settings, setSettings] = useState(preset.settings_json);
  const [visibility, setVisibility] = useState<Visibility>("basic");
  const refresh = () => void client.invalidateQueries({ queryKey: ["presets"] });
  const save = useMutation({ mutationFn: () => api.updatePreset(preset.id, { name, is_default: isDefault, settings }), onSuccess: () => { refresh(); onClose(); } });
  const clone = useMutation({ mutationFn: () => api.clonePreset(preset.id), onSuccess: () => { refresh(); onClose(); } });
  const reset = useMutation({ mutationFn: () => api.resetPreset(preset.id), onSuccess: (value) => { setSettings(value.settings_json); refresh(); } });
  const remove = useMutation({ mutationFn: () => api.deletePreset(preset.id), onSuccess: () => { refresh(); onClose(); } });
  const exportBundle = useMutation({ mutationFn: () => api.exportPreset(preset.id), onSuccess: (bundle) => downloadJson(bundle, `${preset.name.replaceAll(/[^a-z0-9]+/gi, "-").toLowerCase()}.lm-atelier-preset.json`) });
  const engine = engines.find((item) => item.roles.includes(preset.role));
  const fields = resolveCapabilitySettings(engine, preset.role).filter((field) => field.scope !== "load" && visibilityRank[field.visibility] <= visibilityRank[visibility] && field.available);
  const error = save.error ?? clone.error ?? reset.error ?? remove.error ?? exportBundle.error;
  return (
    <AccessibleDialog
      title="Edit preset"
      eyebrow={`${preset.role} generation preset`}
      closeLabel="Close preset editor"
      onClose={onClose}
      className="settings-editor"
    >
      <label>Preset name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label className="toggle-row"><span><strong>Default {preset.role} preset</strong></span><input type="checkbox" checked={isDefault} onChange={(event) => setIsDefault(event.target.checked)} /></label>
      <div className="segmented compact" role="group" aria-label="Preset setting detail">{(["basic", "advanced", "expert"] as Visibility[]).map((level) => <button type="button" key={level} className={visibility === level ? "active" : ""} aria-pressed={visibility === level} onClick={() => setVisibility(level)}>{level}</button>)}</div>
      <div className="settings-list embedded">{fields.map((field) => <div className="scoped-setting" key={`${field.scope}:${field.key}:${JSON.stringify(settings[field.key])}`}><span className="scope-label">{field.scope}</span><SettingControl field={field} value={settings[field.key] ?? field.default} onChange={(value) => setSettings({ ...settings, [field.key]: value })} /></div>)}</div>
      {error && <ErrorCallout message={error.message} />}
      <footer className="editor-actions"><button className="secondary danger" onClick={() => remove.mutate()}>Delete</button><button className="secondary" onClick={() => reset.mutate()}>Reset</button><button className="secondary" onClick={() => exportBundle.mutate()}>Export</button><button className="secondary" onClick={() => clone.mutate()}>Clone</button><button className="primary" onClick={() => save.mutate()} disabled={!name.trim() || save.isPending}>Save preset</button></footer>
    </AccessibleDialog>
  );
}

export function SettingsView({ engines }: { engines: EngineCapabilities[] }) {
  const [confirmDialog, confirm] = useConfirm();
  const client = useQueryClient();
  const [selectedProfile, setSelectedProfile] = useState<ModelProfile | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<GenerationPreset | null>(null);
  const [presetName, setPresetName] = useState("");
  const [presetRole, setPresetRole] = useState<GenerationPreset["role"]>("chat");
  const [importError, setImportError] = useState("");
  const [backupFeedback, setBackupFeedback] = useState<{
    kind: "success" | "error";
    message: string;
  } | null>(null);
  const profileImport = useRef<HTMLInputElement>(null);
  const presetImport = useRef<HTMLInputElement>(null);
  const system = useQuery({ queryKey: ["system"], queryFn: api.system });
  const about = useQuery({ queryKey: ["about"], queryFn: api.about });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const workers = useQuery({ queryKey: ["workers"], queryFn: api.workers, refetchInterval: 3_000 });
  const runtimes = useQuery({
    queryKey: ["runtimes"],
    queryFn: api.runtimes,
    refetchInterval: (query) => query.state.data?.some((runtime) => runtime.state === "installing") ? 2_000 : false,
  });
  const backups = useQuery({ queryKey: ["backups"], queryFn: api.backups });
  const refreshWorkers = () => void client.invalidateQueries({ queryKey: ["workers"] });
  const loadChat = useMutation({ mutationFn: api.loadChatWorker, onSettled: refreshWorkers });
  const startMedia = useMutation({ mutationFn: api.startMediaWorker, onSettled: refreshWorkers });
  const stopWorker = useMutation({ mutationFn: api.stopWorker, onSettled: refreshWorkers });
  const installRuntime = useMutation({
    mutationFn: (engine: RuntimeStatus["engine"]) => api.installRuntime(engine),
    onSuccess: (value: RuntimeStatus) => {
      client.setQueryData<RuntimeStatus[]>(["runtimes"], (current = []) =>
        current.map((item) => item.engine === value.engine ? value : item)
      );
    },
  });
  const storeBackup = (backup: BackupInfo) => {
    client.setQueryData<BackupInfo[]>(["backups"], (current = []) => {
      const existing = current.find((item) => item.name === backup.name);
      const stored = backup.restore_pending
        ? backup
        : { ...backup, restore_pending: existing?.restore_pending ?? false };
      const remaining = current
        .filter((item) => item.name !== backup.name)
        .map((item) => backup.restore_pending ? { ...item, restore_pending: false } : item);
      return [stored, ...remaining].sort(
        (left, right) => Date.parse(right.created_at) - Date.parse(left.created_at),
      );
    });
  };
  const showBackupError = (error: Error) =>
    setBackupFeedback({ kind: "error", message: error.message });
  const createBackup = useMutation({
    mutationFn: api.createBackup,
    onMutate: () => setBackupFeedback(null),
    onSuccess: (backup) => {
      storeBackup(backup);
      setBackupFeedback({ kind: "success", message: "Backup created." });
    },
    onError: showBackupError,
  });
  const verifyBackup = useMutation({
    mutationFn: api.verifyBackup,
    onMutate: () => setBackupFeedback(null),
    onSuccess: (backup) => {
      storeBackup(backup);
      setBackupFeedback({ kind: "success", message: "Backup verified." });
    },
    onError: showBackupError,
  });
  const restoreBackup = useMutation({
    mutationFn: api.restoreBackup,
    onMutate: () => setBackupFeedback(null),
    onSuccess: (backup) => {
      storeBackup(backup);
      setBackupFeedback({
        kind: "success",
        message: "Restore scheduled. Restart LM Atelier to apply this backup.",
      });
    },
    onError: showBackupError,
  });
  const deleteBackup = useMutation({
    mutationFn: api.deleteBackup,
    onMutate: () => setBackupFeedback(null),
    onSuccess: (_value, name) => {
      client.setQueryData<BackupInfo[]>(["backups"], (current = []) =>
        current.filter((backup) => backup.name !== name)
      );
      setBackupFeedback({ kind: "success", message: "Backup deleted." });
    },
    onError: showBackupError,
  });
  const toolProbe = useMutation({ mutationFn: api.probeChatTools });
  const chatWorker = workers.data?.find((worker) => worker.name === "chat");
  const chatWorkerBusy = Boolean(
    chatWorker && chatWorker.active_jobs + chatWorker.queued_jobs > 0
  );
  const createPreset = useMutation({
    mutationFn: () => api.createPreset(presetRole, presetName),
    onSuccess: () => {
      setPresetName("");
      void client.invalidateQueries({ queryKey: ["presets"] });
    },
  });
  const setDefaultProfile = useMutation({
    mutationFn: (profile: ModelProfile) =>
      api.updateProfile(profile.id, { is_default: true }),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["profiles"] }),
  });
  const importBundle = async (file: File | undefined, kind: "profile" | "preset") => {
    if (!file) return;
    setImportError("");
    try {
      const value = JSON.parse(await file.text()) as ModelProfileBundle | GenerationPresetBundle;
      if (kind === "profile") {
        if (value.format !== "lm-atelier-profile") throw new Error("This is not an LM Atelier profile bundle.");
        await api.importProfile(value as ModelProfileBundle);
        await client.invalidateQueries({ queryKey: ["profiles"] });
      } else {
        if (value.format !== "lm-atelier-preset") throw new Error("This is not an LM Atelier preset bundle.");
        await api.importPreset(value as GenerationPresetBundle);
        await client.invalidateQueries({ queryKey: ["presets"] });
      }
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Could not import the bundle.");
    }
  };
  const technicalDetails = about.data && system.data
    ? formatTechnicalDetails(about.data, system.data, engines)
    : "";
  return (
    <div className="page-view settings-page">
      <header className="page-header"><div><h1>Settings</h1></div></header>
      <CredentialSettingsCard
        provider="huggingface"
        providerLabel="Hugging Face"
        description="Access private and gated models. The token stays in your operating-system credential vault and is never displayed after saving."
        environmentVariable="LOCAL_LM_HF_TOKEN"
        placeholder="hf_..."
      />
      <CredentialSettingsCard
        provider="civitai"
        providerLabel="CivitAI"
        description="Store a CivitAI API token securely for authenticated catalog downloads."
        environmentVariable="LOCAL_LM_CIVITAI_TOKEN"
        placeholder="CivitAI API token"
      />
      <section><h2>Engines</h2><div className="engine-grid">{engines.map((engine) => <article className="engine-card" key={`${engine.engine}:${engine.roles.join()}`}><header><div className="model-icon"><Cpu /></div><div><h3>{engine.engine}</h3><p>{engine.roles.join(" · ")} · {engine.version}</p></div><StatusDot healthy={engine.healthy} label={`${engine.engine} engine`} /></header>{engine.roles.includes("chat") && <div className="capability-list"><button className="secondary compact-button" onClick={() => toolProbe.mutate()} disabled={toolProbe.isPending}>{toolProbe.isPending ? "Testing…" : "Test structured tools"}</button></div>}</article>)}</div>{toolProbe.data && <div className={`callout ${toolProbe.data.passed ? "success" : "error"}`} role={toolProbe.data.passed ? "status" : "alert"}>{toolProbe.data.passed ? `Structured tool schema passed on ${toolProbe.data.engine} ${toolProbe.data.version}.` : `Structured tool schema failed: ${toolProbe.data.error || "unknown response"}`}</div>}{toolProbe.error && <ErrorCallout message={toolProbe.error.message} />}<div className="runtime-setup-grid">{runtimes.data?.map((runtime) => <RuntimeSetupCard key={runtime.engine} runtime={runtime} installPending={installRuntime.isPending} onInstall={(engine) => installRuntime.mutate(engine)} />)}</div>{(runtimes.error || installRuntime.error) && <ErrorCallout message={(runtimes.error || installRuntime.error)?.message} />}</section>
      <section><h2>Machine</h2>{system.data && <div className="metric-grid"><div className="cpu-metric"><Cpu /><span><strong>{system.data.cpu_model}</strong><small>CPU model</small></span></div><div><HardDrive /><span><strong>{formatBytes(system.data.disk_free_bytes)}</strong> disk free</span></div></div>}<div className="device-list">{system.data?.devices.filter((device) => device.kind !== "cpu").map((device) => <div key={device.id}><span className="device-icon"><Cpu size={18} /></span><span><strong>{device.name}</strong><small>{device.backend}</small></span></div>)}</div></section>
      <section>
        <div className="detail-title"><div><h2>Model profiles</h2></div><button className="secondary" onClick={() => profileImport.current?.click()}>Import profile</button></div>
        <input ref={profileImport} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0], "profile"); event.target.value = ""; }} />
        <div className="profile-table interactive">{profiles.data?.map((profile: ModelProfile) => <div key={profile.id}><span className="badge">{profile.role}</span><strong>{profile.name}{profile.is_default ? " · default" : ""}</strong><span title={profile.use_case}>{profile.use_case || "No Auto use case yet"}</span><span className="row-actions">{!profile.is_default && <button className="secondary compact-button" aria-label={`Set ${profile.name} as default ${profile.role} model`} disabled={setDefaultProfile.isPending} onClick={() => setDefaultProfile.mutate(profile)}>Set default</button>}{profile.role === "chat" && profile.model_install_id && <button className="secondary compact-button" aria-label={`Load profile: ${profile.name}`} disabled={chatWorkerBusy || loadChat.isPending} title={chatWorkerBusy ? "Wait for active and queued jobs before changing the worker" : "Load this chat profile"} onClick={() => loadChat.mutate(profile.id)}>Load</button>}<button className="secondary compact-button" aria-label={`Edit profile: ${profile.name}`} onClick={() => setSelectedProfile(profile)}>Edit</button></span></div>)}</div>
        {setDefaultProfile.error && <ErrorCallout message={setDefaultProfile.error.message} />}
      </section>
      <section>
        <div className="detail-title"><div><h2>Generation presets</h2><p>Reuse response length, sampling, image size, video length, and seed settings.</p></div><button className="secondary" onClick={() => presetImport.current?.click()}>Import preset</button></div>
        <input ref={presetImport} hidden type="file" accept="application/json,.json" onChange={(event) => { void importBundle(event.target.files?.[0], "preset"); event.target.value = ""; }} />
        <div className="preset-create"><input aria-label="New preset name" placeholder="New preset name" value={presetName} onChange={(event) => setPresetName(event.target.value)} /><select aria-label="New preset role" value={presetRole} onChange={(event) => setPresetRole(event.target.value as GenerationPreset["role"])}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select><button className="primary" disabled={!presetName.trim() || createPreset.isPending} onClick={() => createPreset.mutate()}><Plus size={15} />Create preset</button></div>
        <div className="profile-table interactive">{presets.data?.map((preset) => <div key={preset.id}><span className="badge">{preset.role}</span><strong>{preset.name}{preset.is_default ? " · default" : ""}</strong><span>{Object.keys(preset.settings_json).length} overrides</span><button className="secondary compact-button" aria-label={`Edit preset: ${preset.name}`} onClick={() => setSelectedPreset(preset)}>Edit</button></div>)}</div>
        {(createPreset.error || importError) && <ErrorCallout message={createPreset.error?.message || importError} />}
      </section>
      <section>
        <div className="detail-title">
          <div><h2>Workers</h2><p>A model must finish loading within the startup time limit. Large models on slow disks can need more than the default 60 seconds.</p></div>
          <div className="row-actions">
            <WorkerLogFolderButton />
            <WorkerStartupLimit />
          </div>
        </div>
        <div className="engine-grid">
          {workers.data?.map((worker) => (
            <WorkerStatusCard
              key={worker.name}
              worker={worker}
              startPending={startMedia.isPending}
              stopPending={stopWorker.isPending}
              onStart={() => startMedia.mutate()}
              onStop={() => stopWorker.mutate(worker.name)}
            />
          ))}
        </div>
        <ErrorCallout
          message={(loadChat.error || startMedia.error || stopWorker.error)?.message}
        />
      </section>
      <section>
        <div className="detail-title">
          <div><h2>Recovery backups</h2></div>
          <div className="row-actions">
            <DownloadDiagnosticsButton />
            <button
              className="secondary"
              disabled={createBackup.isPending}
              onClick={() => createBackup.mutate(false)}
            >
              {createBackup.isPending && createBackup.variables === false ? "Backing up…" : "Back up state"}
            </button>
            <button
              className="secondary"
              disabled={createBackup.isPending}
              onClick={() => createBackup.mutate(true)}
            >
              {createBackup.isPending && createBackup.variables === true ? "Backing up…" : "Back up with media"}
            </button>
          </div>
        </div>
        {backups.data?.some((backup) => backup.restore_pending) && (
          <div className="callout success" role="status">
            Restore scheduled. Restart LM Atelier to apply the selected backup.
          </div>
        )}
        {backups.data?.length ? (
          <div className="backup-list">
            {backups.data.map((backup) => {
              const verifying = verifyBackup.isPending && verifyBackup.variables === backup.name;
              const restoring = restoreBackup.isPending && restoreBackup.variables === backup.name;
              const deleting = deleteBackup.isPending && deleteBackup.variables === backup.name;
              return (
                <article className="backup-row" key={backup.name}>
                  <div className="backup-copy">
                    <strong>{formatDate(backup.created_at)}</strong>
                    <small>
                      {formatBytes(backup.size_bytes + backup.media_size_bytes)}
                      {" · "}
                      {backup.media_included ? "State + media" : "State only"}
                      {" · "}
                      {backup.verified ? "Verified" : "Not verified"}
                    </small>
                    <code title={backup.name}>{backup.name}</code>
                  </div>
                  <div className="row-actions">
                    <button
                      className="secondary compact-button"
                      aria-label={`Verify backup ${backup.name}`}
                      disabled={verifying || restoring || deleting}
                      onClick={() => verifyBackup.mutate(backup.name)}
                    >
                      {verifying ? "Verifying…" : "Verify"}
                    </button>
                    <button
                      className="secondary compact-button"
                      aria-label={`Restore backup ${backup.name} on restart`}
                      disabled={backup.restore_pending || verifying || restoring || deleting}
                      onClick={() => void confirm({ title: "Restore this backup on restart?", question: "The next time LM Atelier starts it will replace the current data with this backup. Anything created since the backup was taken is lost.", confirmLabel: "Restore on restart" }).then((ok) => ok && restoreBackup.mutate(backup.name))}
                    >
                      {restoring ? "Scheduling…" : backup.restore_pending ? "Restore scheduled" : "Restore on restart"}
                    </button>
                    <button
                      className="secondary compact-button danger"
                      aria-label={`Delete backup ${backup.name}`}
                      disabled={backup.restore_pending || verifying || restoring || deleting}
                      onClick={() => void confirm({ title: `Delete backup ${backup.name}?`, question: "This is a recovery point. Deleting it cannot be undone and it cannot be recreated from the current data.", confirmLabel: "Delete backup" }).then((ok) => ok && deleteBackup.mutate(backup.name))}
                    >
                      {deleting ? "Deleting…" : "Delete"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          !backups.isLoading && <p className="muted">No recovery backups yet.</p>
        )}
        {backupFeedback && (
          <div
            className={`callout ${backupFeedback.kind}`}
            role={backupFeedback.kind === "error" ? "alert" : "status"}
          >
            {backupFeedback.message}
          </div>
        )}
        {backups.error && <ErrorCallout message={backups.error.message} />}
      </section>
      <section>
        <div className="detail-title">
          <div><h2>About &amp; support</h2></div>
          {about.data && <span className="badge">Version {about.data.version}</span>}
        </div>
        {about.data && <div className="about-support">
          <div className="about-paths">
            <div><Folder size={17} /><span><small>Data folder</small><code>{about.data.data_directory}</code></span><CopyTextButton text={about.data.data_directory} label="Copy data folder" buttonText="Copy data folder" className="secondary compact-button" /></div>
            <div><Folder size={17} /><span><small>Log folder</small><code>{about.data.log_directory}</code></span><CopyTextButton text={about.data.log_directory} label="Copy log folder" buttonText="Copy log folder" className="secondary compact-button" /></div>
            <div><Folder size={17} /><span><small>Artifact folder</small><code>{about.data.artifact_directory}</code></span><CopyTextButton text={about.data.artifact_directory} label="Copy artifact folder" buttonText="Copy artifact folder" className="secondary compact-button" /></div>
            {about.data.artifact_directory_requested && (
              <div><Folder size={17} /><span><small>Artifact folder requested as</small><code>{about.data.artifact_directory_requested}</code></span></div>
            )}
          </div>
          <div className="about-actions">
            {technicalDetails && <CopyTextButton text={technicalDetails} label="Copy technical details" buttonText="Copy technical details" className="secondary" />}
            <nav aria-label="Support resources">
              {supportLinks(about.data.version).map(([label, href]) => (
                <a key={label} href={href} target="_blank" rel="noreferrer">{label}</a>
              ))}
            </nav>
          </div>
        </div>}
        {(about.error || system.error) && <ErrorCallout message="About information is unavailable." />}
      </section>
      {selectedProfile && <ProfileEditor profile={selectedProfile} engines={engines} onClose={() => setSelectedProfile(null)} />}
      {selectedPreset && <PresetEditor preset={selectedPreset} engines={engines} onClose={() => setSelectedPreset(null)} />}
      {confirmDialog}
    </div>
  );
}
