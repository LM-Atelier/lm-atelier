import type { ReactNode } from "react";
import { Search } from "lucide-react";
import { ErrorCallout } from "./ErrorCallout";
import type { useProfileIdentity, useProfileLibrary } from "./useProfileLibrary";

export function ProfileReadStatus({ read }: { read: ReturnType<typeof useProfileIdentity> }) {
  if (read.ready) return null;
  return read.pending ? <p role="status">Loading selected model settings…</p>
    : <ErrorCallout message={read.error?.message ?? "The selected model profile is unavailable."}
      action={<button type="button" onClick={read.retry}>Retry model settings</button>} />;
}

export function ProfilePicker({ library, label, searchLabel, value, selectedId, onChange, children, disabled = false }: {
  library: ReturnType<typeof useProfileLibrary>;
  label: string; searchLabel: string; value: string; selectedId: string | null | undefined;
  onChange: (value: string) => void; children: ReactNode; disabled?: boolean;
}) {
  const { pages, identity } = library;
  const missing = selectedId && selectedId !== "__auto__" && !library.choices.some((profile) => profile.id === selectedId);
  return <div>
    <label>{label}<select aria-label={label} value={value} aria-disabled={disabled}
      onChange={(event) => { if (!disabled) onChange(event.target.value); }}>
      {children}
      {missing && <option value={value} disabled>{identity.pending ? "Selected model (loading…)" : "Selected model unavailable"}</option>}
      {library.choices.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}
    </select></label>
    <label className="search-box"><Search size={18} /><input aria-label={searchLabel} maxLength={500} value={library.search}
      onChange={(event) => library.setSearch(event.target.value)} /></label>
    {pages.isPending && <p role="status">Loading model choices…</p>}
    <ErrorCallout message={pages.error?.message} action={<button type="button" aria-disabled={pages.isFetching}
      onClick={() => { if (!pages.isFetching) void (pages.isFetchNextPageError ? pages.fetchNextPage() : pages.refetch()); }}>Retry model choices</button>} />
    {pages.hasNextPage && <button type="button" aria-disabled={pages.isFetching}
      onClick={() => { if (!pages.isFetching) void pages.fetchNextPage(); }}>{pages.isFetchingNextPage ? "Loading more models…" : "More models"}</button>}
    <ProfileReadStatus read={identity} />
  </div>;
}
