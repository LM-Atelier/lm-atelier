import type { useMediaOrganization } from "./useMediaOrganization";

export function MediaOrganizationCatalogBrowser({ organization, busy = false }: {
  organization: ReturnType<typeof useMediaOrganization>; busy?: boolean;
}) {
  return <details className="media-organization-catalog"><summary>Find albums and tags</summary><div className="media-toolbar">
    {([["albums", organization.albumCatalog], ["tags", organization.tagCatalog]] as const).map(([kind, catalog]) =>
      <fieldset key={kind}>
        <legend>{kind === "albums" ? "Album choices" : "Tag choices"}</legend>
        <label>Search {kind}<input value={catalog.query} maxLength={200} readOnly={busy}
          onChange={(event) => catalog.search(event.target.value)} /></label>
        <div className="row-actions">
          <button className="secondary compact-button" aria-disabled={busy || catalog.pending || !catalog.hasPrevious}
            onClick={() => { if (!busy) catalog.previous(); }}>Previous {kind}</button>
          <button className="secondary compact-button" aria-disabled={busy || catalog.pending || !catalog.hasNext}
            onClick={() => { if (!busy) catalog.next(); }}>Next {kind}</button>
        </div>
      </fieldset>)}
    <button className="secondary" aria-disabled={busy}
      onClick={() => { if (!busy) organization.refresh(); }}>Refresh choices</button>
  </div></details>;
}
