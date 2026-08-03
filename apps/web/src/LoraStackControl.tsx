import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Plus, X } from "lucide-react";
import { api } from "./api";

type LoraSetting = {
  asset_id: string;
  model_strength: number;
  clip_strength: number;
  enabled: boolean;
};

export function LoraStackControl({
  value,
  onChange,
}: {
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const assets = useQuery({
    queryKey: ["model-assets", "lora"],
    queryFn: () => api.modelAssets("lora"),
  });
  const stack: LoraSetting[] = Array.isArray(value)
    ? value.map((item) => (
      item && typeof item === "object" && typeof (item as Record<string, unknown>).asset_id === "string"
        ? {
            asset_id: String((item as Record<string, unknown>).asset_id),
            model_strength: Number((item as Record<string, unknown>).model_strength ?? 1),
            clip_strength: Number((item as Record<string, unknown>).clip_strength ?? 1),
            enabled: (item as Record<string, unknown>).enabled !== false,
          }
        : {
            asset_id: "",
            model_strength: 1,
            clip_strength: 1,
            enabled: false,
          }
    ))
    : [];
  const installed = assets.data?.filter((asset) => asset.active && asset.verified_at) ?? [];
  const used = new Set(stack.map((item) => item.asset_id));
  const addable = installed.find((asset) => !used.has(asset.id));
  const update = (index: number, patch: Partial<LoraSetting>) => {
    onChange(stack.map((item, candidate) => candidate === index ? { ...item, ...patch } : item));
  };
  const move = (index: number, offset: number) => {
    const next = [...stack];
    const target = index + offset;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    onChange(next);
  };
  return (
    <div className="setting-row lora-stack-control">
      <span><strong>LoRAs</strong></span>
      <div className="lora-stack">
        {stack.map((item, index) => {
          const asset = installed.find((candidate) => candidate.id === item.asset_id);
          const metadata = asset?.manifest_json.metadata;
          const triggerWords = metadata && typeof metadata === "object"
            && Array.isArray((metadata as Record<string, unknown>).trigger_words)
            ? (metadata as Record<string, unknown>).trigger_words as string[]
            : [];
          return (
            <div className={`lora-stack-item${asset ? "" : " unavailable"}`} key={`${item.asset_id}:${index}`}>
              <select
                aria-label={`LoRA ${index + 1}`}
                value={item.asset_id}
                onChange={(event) => update(index, { asset_id: event.target.value })}
              >
                {!asset && <option value={item.asset_id}>{item.asset_id ? "Unavailable LoRA" : "Choose LoRA"}</option>}
                {installed.map((candidate) => (
                  <option value={candidate.id} key={candidate.id}>{candidate.name}</option>
                ))}
              </select>
              <label>Model<input aria-label={`LoRA ${index + 1} model strength`} type="number" min="-4" max="4" step="0.05" value={item.model_strength} onChange={(event) => update(index, { model_strength: Number(event.target.value) })} /></label>
              <label>CLIP<input aria-label={`LoRA ${index + 1} CLIP strength`} type="number" min="-4" max="4" step="0.05" value={item.clip_strength} onChange={(event) => update(index, { clip_strength: Number(event.target.value) })} /></label>
              <label className="lora-enabled"><input aria-label={`Enable LoRA ${index + 1}`} type="checkbox" checked={item.enabled} onChange={(event) => update(index, { enabled: event.target.checked })} />On</label>
              <span className="row-actions">
                <button type="button" className="secondary compact-button" aria-label={`Move LoRA ${index + 1} up`} disabled={index === 0} onClick={() => move(index, -1)}><ArrowUp size={13} /></button>
                <button type="button" className="secondary compact-button" aria-label={`Move LoRA ${index + 1} down`} disabled={index === stack.length - 1} onClick={() => move(index, 1)}><ArrowDown size={13} /></button>
                <button type="button" className="secondary compact-button danger" aria-label={`Remove LoRA ${index + 1}`} onClick={() => onChange(stack.filter((_, candidate) => candidate !== index))}><X size={13} /></button>
              </span>
              {(asset?.family || triggerWords.length > 0) && <small>{[asset?.family, ...triggerWords.slice(0, 3)].filter(Boolean).join(" · ")}</small>}
            </div>
          );
        })}
        <button
          type="button"
          className="secondary compact-button"
          disabled={!addable || stack.length >= 8}
          onClick={() => addable && onChange([
            ...stack,
            {
              asset_id: addable.id,
              model_strength: 1,
              clip_strength: 1,
              enabled: true,
            },
          ])}
        >
          <Plus size={13} /> Add LoRA
        </button>
      </div>
    </div>
  );
}
