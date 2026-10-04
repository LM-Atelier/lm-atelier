import { useState } from "react";
import { MediaOrganizationCatalogBrowser } from "./MediaOrganizationCatalogBrowser";
import type { ArtifactLibraryFilters } from "./artifactLibraryPage";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import type { MediaAlbum, MediaTag, OrganizationCommand, SelectionAction } from "./mediaOrganization";
import { type useMediaOrganization } from "./useMediaOrganization";

export function MediaOrganizationControls({ filters, onFilters, organization }: {
  filters: ArtifactLibraryFilters; onFilters: (next: ArtifactLibraryFilters) => void;
  organization: ReturnType<typeof useMediaOrganization>;
}) {
  const [action, setAction] = useState<SelectionAction | "set-favorite" | "unfavorite" | "trash">("set-favorite");
  const [destination, setDestination] = useState("");
  const [chosenTarget, setChosenTarget] = useState<{ id: string; version: number; label: string }>();
  const [filterAlbum, setFilterAlbum] = useState<MediaAlbum>();
  const [filterTag, setFilterTag] = useState<MediaTag>();
  const albumChoices = filterAlbum && filterAlbum.id === filters.collection_id
    && !organization.albums.some((item) => item.id === filterAlbum.id) ? [...organization.albums, filterAlbum] : organization.albums;
  const tagChoices = filterTag && filterTag.id === filters.tag_id
    && !organization.tags.some((item) => item.id === filterTag.id) ? [...organization.tags, filterTag] : organization.tags;
  const albums = action.includes("album");
  const pageChoices = albums ? organization.albums.map((album) => ({ ...album, label: album.name })) : organization.tags;
  const choices = chosenTarget && !pageChoices.some((item) => item.id === chosenTarget.id)
    ? [...pageChoices, chosenTarget] : pageChoices;
  const target = choices.find((item) => item.id === destination);
  const requiresTarget = action !== "set-favorite" && action !== "unfavorite" && action !== "trash";
  const ready = !organization.pending && organization.selected.length > 0 && (!requiresTarget || target !== undefined);
  const review = () => {
    if (!ready) return;
    const entries = organization.selected.map((entry) => ({ id: entry.id, version: entry.version }));
    let command: OrganizationCommand;
    if (action === "set-favorite" || action === "unfavorite") command = { action: "set-favorite", favorite: action === "set-favorite", entries };
    else if (action === "trash") command = { action, entries };
    else { if (!target) return; command = { action, target: { id: target.id, version: target.version }, entries }; }
    void organization.preview(command, organization.selected, { targetLabel: target?.label });
  };
  return <>
    {!organization.manage && <MediaOrganizationCatalogBrowser organization={organization} busy={organization.pending} />}
    <div className="media-toolbar">
      <select aria-label="Album" value={filters.collection_id ?? ""}
        onChange={(event) => {
          setFilterAlbum(albumChoices.find((item) => item.id === event.target.value));
          onFilters({ ...filters, collection_id: event.target.value || undefined });
        }}>
        <option value="">All albums</option>
        {albumChoices.map((album) => <option key={album.id} value={album.id}>{album.name}</option>)}
      </select>
      <select aria-label="Tag" value={filters.tag_id ?? ""}
        onChange={(event) => {
          setFilterTag(tagChoices.find((item) => item.id === event.target.value));
          onFilters({ ...filters, tag_id: event.target.value || undefined });
        }}>
        <option value="">All tags</option>
        {tagChoices.map((tag) => <option key={tag.id} value={tag.id}>{tag.label}</option>)}
      </select>
      <button className="secondary" onClick={() => organization.setManage(true)}>Albums and tags</button>
    </div>
    {organization.catalogFailed && <ErrorCallout message="Albums and tags could not be loaded. Refresh and try again." />}
    {organization.reviewError && <ErrorCallout message={organization.reviewError} />}
    {organization.selected.length > 0 && <div className="media-toolbar">
      <span>{organization.selected.length} selected · {formatBytes(organization.selected.reduce((sum, entry) => sum + entry.size_bytes, 0)).replace(/\.0(?= )/, "")}</span>
      <select aria-label="Selection action" value={action} onChange={(event) => {
        setAction(event.target.value as typeof action); setDestination(""); setChosenTarget(undefined);
      }}>
        <option value="set-favorite">Favorite</option><option value="unfavorite">Remove favorite</option>
        <option value="add-to-album">Add to album</option><option value="remove-from-album">Remove from album</option>
        <option value="reorder-album">Reorder in album</option><option value="add-tag">Add tag</option>
        <option value="remove-tag">Remove tag</option><option value="trash">Move to Recently Deleted</option>
      </select>
      {requiresTarget && <select aria-label={albums ? "Destination album" : "Destination tag"} value={destination}
        onChange={(event) => {
          setDestination(event.target.value); setChosenTarget(choices.find((item) => item.id === event.target.value));
        }}>
        <option value="">Choose {albums ? "an album" : "a tag"}</option>
        {choices.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select>}
      <button className="primary" aria-disabled={!ready} onClick={review}>{organization.pending ? "Reviewing…" : "Review selection"}</button>
      <button className="secondary" aria-disabled={organization.pending} onClick={organization.clear}>Clear selection</button>
      {organization.selected.length === 100 && <span>Select up to 100 items at a time.</span>}
    </div>}
    {action === "reorder-album" && organization.selected.length > 0 && <>
      <p>These selected items will occupy their current album positions in the order below.</p>
      <ol>{organization.selected.map((entry, index) => <li key={entry.id}>
        {entry.display_name}
        <button className="secondary compact-button" aria-label={`Move ${entry.display_name} earlier`}
          aria-disabled={organization.pending || index === 0}
          onClick={() => organization.move(entry.id, -1)}>Move up</button>
        <button className="secondary compact-button" aria-label={`Move ${entry.display_name} later`}
          aria-disabled={organization.pending || index === organization.selected.length - 1}
          onClick={() => organization.move(entry.id, 1)}>Move down</button>
      </li>)}</ol>
    </>}
  </>;
}
