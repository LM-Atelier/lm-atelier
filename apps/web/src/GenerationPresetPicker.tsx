import type { ReactNode } from "react";
import { Search } from "lucide-react";
import { ErrorCallout } from "./ErrorCallout";
import type { useGenerationPresetLibrary } from "./useGenerationPresetLibrary";

export function GenerationPresetPicker({ library, label, searchLabel, value, selectedId = value, onChange, children, disabled = false }: {
  library: ReturnType<typeof useGenerationPresetLibrary>;
  label: string;
  searchLabel: string;
  value: string;
  selectedId?: string | null;
  onChange: (value: string) => void;
  children: ReactNode;
  disabled?: boolean;
}) {
  const { pages } = library;
  const selectedMissing = Boolean(selectedId) && !library.choices.some((preset) => preset.id === selectedId);
  return <div>
    <label className="setting-row"><span><strong>Preset</strong></span>
      <select aria-label={label} value={value} aria-disabled={disabled} onChange={(event) => { if (!disabled) onChange(event.target.value); }}>
        {children}
        {selectedMissing && <option value={value} disabled>{library.pending ? "Selected preset (loading…)" : "Selected preset unavailable"}</option>}
        {library.choices.map((preset) => <option key={preset.id} value={preset.id}>{preset.name}</option>)}
      </select>
    </label>
    <label className="search-box"><Search size={18} /><input aria-label={searchLabel} maxLength={500} value={library.search} onChange={(event) => library.setSearch(event.target.value)} /></label>
    {pages.isPending && <p role="status">Loading preset choices…</p>}
    <ErrorCallout message={pages.error?.message} action={<button type="button" className="secondary compact-button" aria-disabled={pages.isFetching} onClick={() => { if (!pages.isFetching) void (pages.isFetchNextPageError ? pages.fetchNextPage() : pages.refetch()); }}>Retry preset choices</button>} />
    {pages.hasNextPage && <div className="load-more"><button type="button" className="secondary" aria-disabled={pages.isFetching} onClick={() => { if (!pages.isFetching) void pages.fetchNextPage(); }}>{pages.isFetchingNextPage ? "Loading more presets…" : "More presets"}</button></div>}
  </div>;
}
