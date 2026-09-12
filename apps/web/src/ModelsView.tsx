import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Folder,
  HardDrive,
  Search,
} from "lucide-react";
import { useMemo, useState } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { FirstFailure } from "./FirstFailure";
import { InstallConfirmDialog } from "./InstallConfirmDialog";
import { ModelCard } from "./ModelCard";
import { ModelUpdatesPanel } from "./ModelUpdatesPanel";
import { RecipeCard } from "./RecipeCard";
import { VersionChooser } from "./VersionChooser";
import { WorkflowConsumers } from "./WorkflowConsumers";
import { api } from "./api";
import { formatBytes } from "./format";
import type {
  CatalogModel,
  CatalogPreflight,
  EngineRole,
  ModelAssetInstall,
  ModelInstall,
  ModelProfile,
} from "./types";
import { useConfirm } from "./useConfirm";

function InstalledModelRow({
  model,
  profile,
  creating,
  deleting,
  saving,
  defaulting,
  onCreate,
  onDelete,
  onSaveUseCase,
  onSetDefault,
}: {
  model: ModelInstall;
  profile?: ModelProfile;
  creating: boolean;
  deleting: boolean;
  saving: boolean;
  defaulting: boolean;
  onCreate: () => void;
  onDelete: () => void;
  onSaveUseCase: (value: string) => Promise<boolean>;
  onSetDefault: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(profile?.use_case ?? "");
  const startEditing = () => {
    setDraft(profile?.use_case ?? "");
    setEditing(true);
  };
  const save = async () => {
    if (await onSaveUseCase(draft.trim())) setEditing(false);
  };
  return (
    <div className={editing ? "editing" : ""}>
      <span className="badge">{model.role}</span>
      <span className="model-install-copy">
        <strong>{model.name}</strong>
        <small>{model.readiness === "ready" ? "Runtime verified" : model.readiness === "unsupported" ? "Unsupported" : "Not runtime verified"}</small>
        {model.role === "chat" && model.readiness === "ready" && profile?.input_modalities?.includes("text") && (
          <small>{profile.input_modalities.includes("image") ? "Vision capable" : "Text only"}</small>
        )}
        {profile?.use_case && <small>{profile.use_case}</small>}
      </span>
      <span className="model-install-size">{formatBytes(model.size_bytes)}</span>
      <span className="row-actions">
        {profile?.is_default
          ? <span className="badge tested">Default</span>
          : <button className="secondary compact-button" aria-label={`Set ${model.name} as default ${model.role} model`} disabled={creating || defaulting} onClick={onSetDefault}>{defaulting ? "Setting..." : "Set default"}</button>}
        {profile
          ? <button className="secondary compact-button" aria-label={`Edit use case for ${model.name}`} onClick={startEditing} disabled={editing || saving}>Edit use case</button>
          : <button className="secondary compact-button" aria-label={`Add ${model.name} to model selectors`} disabled={creating} onClick={onCreate}>Add to selectors</button>}
        <button className="secondary compact-button danger" aria-label={`Delete ${model.name}`} disabled={deleting} onClick={onDelete}>Delete</button>
      </span>
      {editing && profile && (
        <form className="model-use-case-editor" onSubmit={(event) => { event.preventDefault(); void save(); }}>
          <textarea aria-label={`Best uses for ${model.name}`} rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Programming, illustration, cinematic video…" />
          <button type="button" className="secondary compact-button" disabled={saving} onClick={() => setEditing(false)}>Cancel</button>
          <button type="submit" className="primary compact-button" disabled={saving || draft.trim() === profile.use_case.trim()}>{saving ? "Saving…" : "Save"}</button>
        </form>
      )}
    </div>
  );
}

type ModelAssetUpdateValues = Partial<Pick<
  ModelAssetInstall,
  "active" | "use_case" | "auto_apply" | "default_model_strength" | "default_clip_strength"
>>;

function InstalledAssetRow({
  asset,
  saving,
  deleting,
  onUpdate,
  onDelete,
}: {
  asset: ModelAssetInstall;
  saving: boolean;
  deleting: boolean;
  onUpdate: (values: ModelAssetUpdateValues) => Promise<boolean>;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [useCase, setUseCase] = useState(asset.use_case);
  const [autoApply, setAutoApply] = useState(asset.auto_apply);
  const [modelStrength, setModelStrength] = useState(String(asset.default_model_strength));
  const [clipStrength, setClipStrength] = useState(String(asset.default_clip_strength));
  const beginEditing = () => {
    setUseCase(asset.use_case);
    setAutoApply(asset.auto_apply);
    setModelStrength(String(asset.default_model_strength));
    setClipStrength(String(asset.default_clip_strength));
    setEditing(true);
  };
  const parsedModelStrength = Number(modelStrength);
  const parsedClipStrength = Number(clipStrength);
  const strengthsValid = Number.isFinite(parsedModelStrength)
    && Math.abs(parsedModelStrength) <= 4
    && Number.isFinite(parsedClipStrength)
    && Math.abs(parsedClipStrength) <= 4;
  const unchanged = useCase.trim() === asset.use_case
    && autoApply === asset.auto_apply
    && parsedModelStrength === asset.default_model_strength
    && parsedClipStrength === asset.default_clip_strength;
  const save = async () => {
    if (!strengthsValid || (autoApply && !useCase.trim())) return;
    const saved = await onUpdate({
      use_case: useCase.trim(),
      auto_apply: autoApply,
      default_model_strength: parsedModelStrength,
      default_clip_strength: parsedClipStrength,
    });
    if (saved) setEditing(false);
  };
  return (
    <div className={editing ? "editing" : ""}>
      <span className="badge">{asset.kind.replace("_", " ")}</span>
      <span className="model-install-copy">
        <strong>{asset.name}</strong>
        <small>{asset.active ? "Ready" : "Disabled"}{asset.family ? ` · ${asset.family}` : ""}</small>
        {asset.kind === "lora" && asset.auto_apply && asset.use_case && (
          <small>Auto · {asset.use_case}</small>
        )}
      </span>
      <span className="model-install-size">{formatBytes(asset.size_bytes)}</span>
      <span className="row-actions">
        <button
          className="secondary compact-button"
          disabled={!asset.verified_at || saving}
          onClick={() => void onUpdate({ active: !asset.active })}
        >
          {asset.active ? "Disable" : "Enable"}
        </button>
        {asset.kind === "lora" && (
          <button className="secondary compact-button" disabled={editing || saving} onClick={beginEditing}>
            Edit Auto rules
          </button>
        )}
        <button className="secondary compact-button danger" disabled={deleting} onClick={onDelete}>Delete</button>
      </span>
      {editing && asset.kind === "lora" && (
        <form className="model-use-case-editor lora-auto-editor" onSubmit={(event) => { event.preventDefault(); void save(); }}>
          <label>
            Use case
            <textarea aria-label={`Auto use case for ${asset.name}`} rows={2} value={useCase} onChange={(event) => setUseCase(event.target.value)} placeholder="Watercolor landscapes, product photography…" />
          </label>
          <label>
            Model strength
            <input aria-label={`Default model strength for ${asset.name}`} type="number" min="-4" max="4" step="0.05" value={modelStrength} onChange={(event) => setModelStrength(event.target.value)} />
          </label>
          <label>
            CLIP strength
            <input aria-label={`Default CLIP strength for ${asset.name}`} type="number" min="-4" max="4" step="0.05" value={clipStrength} onChange={(event) => setClipStrength(event.target.value)} />
          </label>
          <label className="lora-auto-toggle">
            <input aria-label={`Use ${asset.name} automatically`} type="checkbox" checked={autoApply} onChange={(event) => setAutoApply(event.target.checked)} />
            Use automatically
          </label>
          <span className="row-actions">
            <button type="button" className="secondary compact-button" disabled={saving} onClick={() => setEditing(false)}>Cancel</button>
            <button type="submit" className="primary compact-button" disabled={saving || unchanged || !strengthsValid || (autoApply && !useCase.trim())}>{saving ? "Saving…" : "Save"}</button>
          </span>
        </form>
      )}
    </div>
  );
}

interface PendingInstall {
  model: CatalogModel;
  preflight: CatalogPreflight;
  installRole: string;
  engine: string;
  auxiliaryKind: "lora" | null;
}

export function ModelsView({ initialRole }: { initialRole: EngineRole }) {
  const [choosingVersions, setChoosingVersions] = useState<CatalogModel | null>(null);
  const [confirmDialog, confirm] = useConfirm();
  const client = useQueryClient();
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [catalogSource, setCatalogSource] = useState("huggingface");
  const [role, setRole] = useState<string>(initialRole);
  const [sort, setSort] = useState("trending");
  const [compatibility, setCompatibility] = useState("");
  const [quantization, setQuantization] = useState("");
  const [maxSizeGb, setMaxSizeGb] = useState("");
  const [updatedWithinDays, setUpdatedWithinDays] = useState("");
  const [installedChatCapability, setInstalledChatCapability] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [importName, setImportName] = useState("");
  const [importPath, setImportPath] = useState("");
  const [importRole, setImportRole] = useState("chat");
  const [importEngine, setImportEngine] = useState("llama.cpp");
  const catalogFilters = {
    compatibility,
    quantization,
    max_size_bytes: maxSizeGb ? String(Number(maxSizeGb) * 1024 ** 3) : "",
    updated_within_days: updatedWithinDays,
  };
  const catalog = useInfiniteQuery({
    queryKey: ["catalog", submitted, role, sort, catalogFilters, catalogSource],
    queryFn: ({ pageParam }) =>
      api.catalog(submitted, role, sort, pageParam, catalogFilters, catalogSource),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const rawCatalogItems = useMemo(() => catalog.data?.pages.flatMap((page) => page.items) ?? [], [catalog.data]);
  const catalogItems = rawCatalogItems;
  const catalogIsStale = catalog.data?.pages.some((page) => page.stale) ?? false;
  const recipes = useQuery({ queryKey: ["recipes"], queryFn: api.recipes });
  const installed = useQuery({ queryKey: ["models"], queryFn: api.models });
  const modelAssets = useQuery({ queryKey: ["model-assets"], queryFn: () => api.modelAssets() });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3_000 });
  const storage = useQuery({ queryKey: ["model-storage"], queryFn: api.modelStorage });
  const profiles = useQuery({ queryKey: ["profiles"], queryFn: api.profiles });
  const runtimes = useQuery({ queryKey: ["runtimes"], queryFn: api.runtimes });
  const machine = useQuery({ queryKey: ["system"], queryFn: api.system });
  const runtimeFor = (model: CatalogModel) => runtimes.data?.find(
    (runtime) => runtime.engine === model.required_runtime,
  );
  // Preflight and transfer are separate steps so the user sees what a download
  // will cost before it starts; the numbers were previously computed and dropped.
  const [pendingInstall, setPendingInstall] = useState<PendingInstall | null>(null);
  const download = useMutation({
    mutationFn: async ({ model, selectedRole }: { model: CatalogModel; selectedRole: string }) => {
      const auxiliaryKind = selectedRole === "lora" ? "lora" : null;
      const installRole = auxiliaryKind ? "image" : selectedRole;
      const engine = model.required_runtime ?? (installRole === "chat" ? "llama.cpp" : "comfyui");
      // A CivitAI card's remote id is its exact version; that is also the
      // revision it pins. Hugging Face keeps floating "main".
      const revision = model.provider === "civitai" ? model.remote_id : "main";
      const preflight = auxiliaryKind
        ? await api.catalogPreflight(
            model.remote_id,
            installRole,
            engine,
            revision,
            [],
            auxiliaryKind,
            null,
            model.provider,
          )
        : await api.catalogPreflight(
            model.remote_id,
            installRole,
            engine,
            revision,
            [],
            null,
            // Preflight the exact workflow this card represents; a repository
            // can ship several and ranking must not answer for the user.
            model.workflow_template_id ?? null,
            model.provider,
          );
      if (!preflight.can_install) {
        const blockers = preflight.checks
          .filter((check) => check.status === "block")
          .map((check) => check.detail);
        throw new Error(blockers.join(" ") || "This model cannot be installed safely.");
      }
      if (!preflight.install_plan || preflight.install_plan.compatibility !== "supported") {
        throw new Error(
          preflight.install_plan?.failure_reason
          || "LM Atelier cannot safely activate this model with the current runtime.",
        );
      }
      return { model, preflight, installRole, engine, auxiliaryKind } satisfies PendingInstall;
    },
    onSuccess: (ready) => setPendingInstall(ready),
  });
  const confirmInstall = useMutation({
    mutationFn: ({ preflight, installRole, engine, auxiliaryKind }: PendingInstall) => {
      const downloadArguments = [
        preflight.remote_id,
        preflight.source_remote_id,
        installRole,
        engine,
        preflight.revision,
        preflight.selected_files,
        preflight.expected_sha256,
        preflight.file_sources ?? {},
        preflight.comfy_paths,
        preflight.workflow_template_id,
        preflight.workflow_template_sha256,
        preflight.install_plan?.id ?? null,
      ] as const;
      const contentRating = preflight.content_rating ?? "unknown";
      return auxiliaryKind
        ? api.download(...downloadArguments, auxiliaryKind, contentRating)
        : api.download(...downloadArguments, null, contentRating);
    },
    onSuccess: () => {
      setPendingInstall(null);
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const installRecipe = useMutation({
    mutationFn: (recipeId: string) => api.installRecipe(recipeId),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const createProfile = useMutation({
    mutationFn: (model: ModelInstall) => api.createProfile(model),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["profiles"] }),
  });
  const updateUseCase = useMutation({
    mutationFn: ({ profileId, useCase }: { profileId: string; useCase: string }) =>
      api.updateProfile(profileId, { use_case: useCase }),
    onSuccess: (updated) => {
      client.setQueryData<ModelProfile[]>(["profiles"], (current) =>
        current?.map((profile) => profile.id === updated.id ? updated : profile) ?? [updated],
      );
      void client.invalidateQueries({ queryKey: ["profiles"] });
    },
  });
  const setDefaultModel = useMutation({
    mutationFn: ({ model, profile }: { model: ModelInstall; profile?: ModelProfile }) =>
      profile
        ? api.updateProfile(profile.id, { is_default: true })
        : api.createProfile(model, true),
    onSuccess: (updated) => {
      client.setQueryData<ModelProfile[]>(["profiles"], (current) => {
        const siblings = (current ?? []).map((profile) => (
          profile.role === updated.role
            ? { ...profile, is_default: profile.id === updated.id }
            : profile
        ));
        return siblings.some((profile) => profile.id === updated.id)
          ? siblings
          : [...siblings, updated];
      });
      void client.invalidateQueries({ queryKey: ["profiles"] });
    },
  });
  const deleteModel = useMutation({
    mutationFn: (modelId: string) => api.deleteModel(modelId, true),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["profiles"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const updateModelAsset = useMutation({
    mutationFn: ({ id, values }: { id: string; values: ModelAssetUpdateValues }) =>
      api.updateModelAsset(id, values),
    onSuccess: (updated) => {
      client.setQueryData<ModelAssetInstall[]>(["model-assets"], (current) =>
        current?.map((asset) => asset.id === updated.id ? updated : asset) ?? [updated],
      );
      void client.invalidateQueries({ queryKey: ["model-assets"] });
    },
  });
  const deleteModelAsset = useMutation({
    mutationFn: api.deleteModelAsset,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["model-assets"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const cleanupDownloads = useMutation({
    mutationFn: api.cleanupDownloads,
    onSuccess: () => void client.invalidateQueries({ queryKey: ["model-storage"] }),
  });
  const importModel = useMutation({
    mutationFn: () => api.importModel({ name: importName, local_path: importPath, role: importRole, engine: importEngine }),
    onSuccess: () => {
      setImportOpen(false);
      setImportName("");
      setImportPath("");
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["model-storage"] });
    },
  });
  const installedRemoteIds = new Set(
    (
      role === "lora"
        ? modelAssets.data
          ?.filter((asset) => asset.kind === "lora" && asset.active)
          .map((asset) => asset.manifest_json.remote_id)
        : installed.data
          ?.filter((model) => model.role === role && model.active)
          ?.flatMap((model) => [
            model.manifest_json.remote_id,
            model.manifest_json.source_remote_id,
          ])
    )?.filter((remoteId): remoteId is string => typeof remoteId === "string") ?? [],
  );
  const activeDownloadIds = new Set(
    jobs.data
      ?.filter((job) =>
        job.kind === "download"
        && (
          role === "lora"
            ? job.payload_json.auxiliary_kind === "lora"
            : job.payload_json.role === role && !job.payload_json.auxiliary_kind
        )
        && ["queued", "running", "paused"].includes(job.status)
      )
      .map((job) => job.payload_json.remote_id)
      .filter((remoteId): remoteId is string => typeof remoteId === "string") ?? [],
  );
  const installedModels = (installed.data ?? []).filter((model) => {
    if (!installedChatCapability) return true;
    if (model.role !== "chat" || model.readiness !== "ready") return false;
    const profile = profiles.data?.find((candidate) => candidate.model_install_id === model.id);
    const modalities = profile?.input_modalities ?? [];
    return installedChatCapability === "vision"
      ? modalities.includes("image")
      : modalities.includes("text") && !modalities.includes("image");
  });
  const installedTemplateIds = new Set(installed.data?.filter((model) => model.role === role && model.active).map((model) => model.manifest_json.workflow_template_id).filter((value): value is string => typeof value === "string") ?? []);
  // A workflow card is installed only when ITS template is: variants share one
  // repository, and installing one must not disable the others.
  const statusFor = (model: CatalogModel): "idle" | "preparing" | "downloading" | "installed" => (
    (model.workflow_template_id ? installedTemplateIds.has(model.workflow_template_id) : installedRemoteIds.has(model.remote_id))
      ? "installed"
      : activeDownloadIds.has(model.remote_id)
        ? "downloading"
        : download.isPending && download.variables?.model.remote_id === model.remote_id && (download.variables?.model.workflow_template_id ?? null) === (model.workflow_template_id ?? null)
          ? "preparing"
          : "idle"
  );
  return (
    <div className="page-view">
      <header className="page-header"><div><h1>Model library</h1></div><div className="storage-actions"><div className="storage-pill"><HardDrive size={17} />{storage.data?.installed_count ?? installed.data?.length ?? 0} installed · {formatBytes(storage.data?.installed_bytes)}</div><button className="secondary compact-button" onClick={() => setImportOpen(true)}><Folder size={16} />Import local</button>{Boolean(storage.data?.partial_download_count) && <button className="secondary compact-button" disabled={cleanupDownloads.isPending} onClick={() => cleanupDownloads.mutate()}>Clean {storage.data?.partial_download_count} partial</button>}</div></header>
      <ModelUpdatesPanel onInstall={(model, selectedRole) => download.mutate({ model, selectedRole })} />
      <section className="recipe-section">
        <div className="section-heading"><div><h2>Reference recipes</h2></div></div>
        {recipes.isLoading && <div className="loading-line" />}
        <FirstFailure of={[recipes, installRecipe]} />
        <div className="recipe-grid">{recipes.data?.map((recipe) => <RecipeCard key={recipe.id} recipe={recipe} pending={installRecipe.isPending && installRecipe.variables === recipe.id} onInstall={() => installRecipe.mutate(recipe.id)} />)}</div>
      </section>
      <div className="toolbar">
        <form className="search-box" onSubmit={(event) => { event.preventDefault(); setSubmitted(query); }}><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models" /></form>
        <select aria-label="Model role" value={role} onChange={(event) => setRole(event.target.value)}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option><option value="lora">LoRA</option></select>
        <select aria-label="Model source" value={catalogSource} onChange={(event) => setCatalogSource(event.target.value)}><option value="huggingface">Hugging Face</option><option value="civitai">CivitAI</option></select><select aria-label="Model order" value={sort} onChange={(event) => setSort(event.target.value)}><option value="trending">Trending</option><option value="downloads">Downloads</option><option value="likes">Likes</option><option value="newest">Newest</option><option value="updated">Recently updated</option><option value="compatible">Compatible first</option></select>
      </div>
      <div className="catalog-filters"><select aria-label="Compatibility filter" value={compatibility} onChange={(event) => setCompatibility(event.target.value)}><option value="">All compatibility</option><option value="likely">Automatic test available</option><option value="advanced_import">Advanced import</option><option value="unsupported">Unsupported</option></select><select aria-label="Last updated filter" value={updatedWithinDays} onChange={(event) => setUpdatedWithinDays(event.target.value)}><option value="">Updated any time</option><option value="7">Updated this week</option><option value="30">Updated this month</option><option value="90">Updated in 3 months</option><option value="365">Updated this year</option></select><input aria-label="Quantization filter" placeholder="Quantization (Q4_K_M, FP8…)" value={quantization} onChange={(event) => setQuantization(event.target.value)} /><input aria-label="Maximum download size" type="number" min="0" placeholder="Max download (GB)" value={maxSizeGb} onChange={(event) => setMaxSizeGb(event.target.value)} /></div>
      {(installed.data?.length ?? 0) > 0 && <section>
        <div className="section-heading">
          <h2>Installed models</h2>
          <select
            aria-label="Installed chat capability"
            value={installedChatCapability}
            onChange={(event) => setInstalledChatCapability(event.target.value)}
          >
            <option value="">All capabilities</option>
            <option value="text">Text only</option>
            <option value="vision">Vision capable</option>
          </select>
        </div>
        <div className="profile-table model-installs">{installedModels.map((model) => {
        const profile = profiles.data?.find((candidate) => candidate.model_install_id === model.id);
        return <InstalledModelRow
          key={model.id}
          model={model}
          profile={profile}
          creating={createProfile.isPending && createProfile.variables?.id === model.id}
          deleting={deleteModel.isPending && deleteModel.variables === model.id}
          saving={updateUseCase.isPending && updateUseCase.variables?.profileId === profile?.id}
          defaulting={setDefaultModel.isPending && setDefaultModel.variables?.model.id === model.id}
          onCreate={() => createProfile.mutate(model)}
          onDelete={() => void confirm({ title: `Delete ${model.name}?`, question: "This removes the model file and its saved settings from local storage. Downloading it again is the only way back.", detail: <WorkflowConsumers kind="model_install" resourceId={model.id} />, confirmLabel: "Delete model" }).then((ok) => ok && deleteModel.mutate(model.id))}
          onSaveUseCase={async (value) => {
            if (!profile) return false;
            try {
              await updateUseCase.mutateAsync({ profileId: profile.id, useCase: value });
              return true;
            } catch {
              return false;
            }
          }}
          onSetDefault={() => setDefaultModel.mutate({ model, profile })}
        />;
        })}</div>
      </section>}
      {(modelAssets.data?.length ?? 0) > 0 && <section>
        <div className="section-heading"><h2>Installed workflow assets</h2></div>
        <div className="profile-table model-installs">
          {modelAssets.data?.map((asset) => (
            <InstalledAssetRow
              key={asset.id}
              asset={asset}
              saving={updateModelAsset.isPending && updateModelAsset.variables?.id === asset.id}
              deleting={deleteModelAsset.isPending && deleteModelAsset.variables === asset.id}
              onUpdate={async (values) => {
                try {
                  await updateModelAsset.mutateAsync({ id: asset.id, values });
                  return true;
                } catch {
                  return false;
                }
              }}
              onDelete={() => void confirm({ title: `Delete ${asset.name}?`, question: "This removes the file from local storage.", detail: <WorkflowConsumers kind="model_asset" resourceId={asset.id} />, confirmLabel: "Delete" }).then((ok) => ok && deleteModelAsset.mutate(asset.id))}
            />
          ))}
        </div>
      </section>}
      {pendingInstall && (
        <InstallConfirmDialog
          name={pendingInstall.model.name || pendingInstall.model.remote_id}
          preflight={pendingInstall.preflight}
          system={machine.data}
          pending={confirmInstall.isPending}
          onConfirm={() => confirmInstall.mutate(pendingInstall)}
          onCancel={() => setPendingInstall(null)}
        />
      )}
      <FirstFailure of={[createProfile, download, confirmInstall, deleteModel, cleanupDownloads, updateUseCase, setDefaultModel, updateModelAsset, deleteModelAsset]} />
      {/* isFetching, not isLoading: the latter is only true the first
          time, so changing a filter swapped the results with no sign
          anything had happened - which reads as the page refreshing
          itself for no reason. */}
      {catalog.isFetching && !catalog.isFetchingNextPage && (
        <div className="catalog-loading" role="status">
          <div className="loading-line" />
          <span>{catalogItems.length > 0 ? "Finding models…" : "Loading the catalogue…"}</span>
        </div>
      )}
      <ErrorCallout message={catalog.error?.message} action={<button className="secondary compact-button" disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>Retry</button>} />
      {catalogIsStale && !catalog.error && <div className="callout warning action-callout" role="status"><span>Showing saved results while Hugging Face is unavailable.</span><button className="secondary compact-button" disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>Refresh</button></div>}
      <div className={`model-grid ${catalog.isFetching && !catalog.isFetchingNextPage ? "superseded" : ""}`}>{catalogItems.map((model) => <ModelCard key={model.remote_id} model={model} role={role} runtime={runtimeFor(model)} status={statusFor(model)} onDownload={() => download.mutate({ model, selectedRole: role })} onChooseVersion={model.provider === "civitai" && model.parent_model_id ? () => setChoosingVersions(model) : undefined} />)}</div>
      {choosingVersions?.parent_model_id && (
        <VersionChooser modelId={choosingVersions.parent_model_id} modelName={choosingVersions.parent_model_name ?? choosingVersions.name}
          onClose={() => setChoosingVersions(null)}
          onChoose={(versionId) => { setChoosingVersions(null); download.mutate({ model: { ...choosingVersions, remote_id: versionId }, selectedRole: role }); }} />
      )}
      {catalog.hasNextPage && <div className="load-more"><button className="secondary" disabled={catalog.isFetchingNextPage} onClick={() => void catalog.fetchNextPage()}>{catalog.isFetchingNextPage ? "Loading…" : "Load more models"}</button></div>}
      {importOpen && (
        <AccessibleDialog
          title="Import a local model"
          eyebrow="Advanced import"
          closeLabel="Close local import"
          onClose={() => setImportOpen(false)}
        >
          <p>Register a local file or folder. Pickle-compatible formats are blocked as unsafe, and imports require review before use.</p>
          <label>Display name<input value={importName} onChange={(event) => setImportName(event.target.value)} /></label>
          <label>Absolute local path<input value={importPath} onChange={(event) => setImportPath(event.target.value)} placeholder="/path/to/model.gguf" /></label>
          <label>Role<select value={importRole} onChange={(event) => { const next = event.target.value; setImportRole(next); setImportEngine(next === "chat" ? "llama.cpp" : "comfyui"); }}><option value="chat">Chat</option><option value="image">Image</option><option value="video">Video</option></select></label>
          <label>Runtime<select value={importEngine} onChange={(event) => setImportEngine(event.target.value)}><option value="llama.cpp">llama.cpp</option><option value="vllm">vLLM (ModelOpt/NVFP4)</option><option value="comfyui">ComfyUI</option></select></label>
          {importModel.error && <ErrorCallout message={importModel.error.message} />}
          <footer><button className="secondary" onClick={() => setImportOpen(false)}>Cancel</button><button className="primary" disabled={!importName.trim() || !importPath.trim() || importModel.isPending} onClick={() => importModel.mutate()}>{importModel.isPending ? "Importing…" : "Import model"}</button></footer>
        </AccessibleDialog>
      )}
      {confirmDialog}
    </div>
  );
}
