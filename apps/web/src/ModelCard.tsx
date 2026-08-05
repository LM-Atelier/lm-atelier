import { Bot, Download, Film, Image as ImageIcon, Sparkles } from "lucide-react";
import { formatBytes, formatDate } from "./format";
import type { CatalogModel, RuntimeStatus } from "./types";

export function ModelCard({
  model,
  role,
  onDownload,
  onChooseVersion,
  status,
  runtime,
}: {
  model: CatalogModel;
  role: string;
  onDownload: () => void;
  /** Opened instead of installing when this card stands for several versions. */
  onChooseVersion?: () => void;
  status: "idle" | "preparing" | "downloading" | "installed";
  runtime?: RuntimeStatus;
}) {
  const label = {
    idle: "Install",
    preparing: "Checking model…",
    downloading: "Downloading…",
    installed: "Installed",
  }[status];
  const compatibilityLabel = {
    likely: "Automatic test available",
    tested: "Tested",
    advanced_import: "Advanced import",
    unsupported: "Unsupported",
  }[model.compatibility] ?? model.compatibility.replace("_", " ");
  const runtimeUnavailable = Boolean(
    model.required_runtime === "vllm" && (!runtime || !runtime.supported),
  );
  const displayCompatibility = model.required_runtime === "vllm"
    ? runtimeUnavailable
      ? "Needs vLLM"
      : "Automatic test available"
    : compatibilityLabel;
  // A card standing for several versions must not install one. The point of
  // version identity is that the person picked it, so the action becomes the
  // chooser and the count says why.
  const versions = model.version_count ?? 1;
  const choosing = versions > 1 && status === "idle" && Boolean(onChooseVersion);
  // How many are already here, when that is knowable at all. `null` is not
  // zero: it means this kind records no provider version, so nothing on disk
  // can be matched against these. Saying "0 of 12" there would be a claim the
  // data cannot support.
  const here = model.installed_version_count;
  const actionLabel = choosing
    ? here == null
      ? `Choose from ${versions} versions`
      : here === 0
        ? `Choose from ${versions} versions`
        : `Manage ${versions} versions`
    : status === "idle" && model.compatibility === "unsupported"
      ? "No workflow"
      : status === "idle" && runtimeUnavailable
        ? "Needs vLLM"
        : label;
  return (
    <article className="model-card">
      <div className="model-icon">{role === "video" ? <Film /> : role === "image" || role === "lora" ? <ImageIcon /> : <Bot />}</div>
      <div className="model-copy">
        <h3>{model.name}</h3><p>{model.author} · {model.pipeline_tag || model.library_name || "model"}</p>
        <div className="badges"><span className={`badge ${model.compatibility}`}>{displayCompatibility}</span>{model.gated && <span className="badge">Gated</span>}{model.formats.slice(0, 2).map((format) => <span className="badge" key={format}>{format}</span>)}{model.quantizations.slice(0, 2).map((value) => <span className="badge" key={value}>{value}</span>)}</div>
        {choosing && here != null && here > 0 && (
          <div className="badges"><span className="badge likely">{here} of {versions} installed</span></div>
        )}
        <small>{model.total_size_bytes != null ? `${formatBytes(model.total_size_bytes)} · ` : ""}{formatDate(model.last_modified)}{model.compatibility_reasons.length ? ` · ${model.compatibility_reasons.join(" · ")}` : ""}</small>
      </div>
      <div className="model-stats">{model.trending_score != null && <span title="Hugging Face trending score"><Sparkles size={14} />{model.trending_score.toLocaleString()}</span>}<span><Download size={14} />{model.downloads?.toLocaleString() ?? "—"}</span><button className="primary compact-button" title={model.compatibility === "unsupported" || runtimeUnavailable ? model.compatibility_reasons.join(" ") : undefined} onClick={choosing ? onChooseVersion : onDownload} disabled={status !== "idle" || model.compatibility === "unsupported" || runtimeUnavailable}>{actionLabel}</button></div>
    </article>
  );
}
