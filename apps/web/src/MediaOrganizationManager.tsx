import { useState } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { MediaOrganizationCatalogBrowser } from "./MediaOrganizationCatalogBrowser";
import { type MediaAlbum, type MediaTag, type OrganizationCommand } from "./mediaOrganization";
import { type OrganizationReviewLabels, type useMediaOrganization } from "./useMediaOrganization";

function OrganizationEditor({ item, tags, pending, onReview }: {
  item: MediaAlbum | MediaTag; tags: MediaTag[]; pending: boolean;
  onReview: (command: OrganizationCommand, labels: OrganizationReviewLabels) => void;
}) {
  const album = "name" in item;
  const [name, setName] = useState(album ? item.name : item.label);
  const [description, setDescription] = useState(album ? item.description : "");
  const [color, setColor] = useState(!album ? item.color ?? "" : "");
  const [destination, setDestination] = useState("");
  const [chosenTag, setChosenTag] = useState<MediaTag>();
  const choices = chosenTag && !tags.some((tag) => tag.id === chosenTag.id) ? [...tags, chosenTag] : tags;
  const target = { id: item.id, version: item.version };
  const targetLabel = album ? item.name : item.label;
  return <section>
    <label>{album ? "Album name" : "Tag label"}<input value={name} maxLength={200}
      onChange={(event) => setName(event.target.value)} /></label>
    {album ? <label>Album description<textarea value={description} maxLength={2000}
      onChange={(event) => setDescription(event.target.value)} /></label>
      : <label>Tag color<input value={color} placeholder="Optional #rrggbb" maxLength={7}
        onChange={(event) => setColor(event.target.value)} /></label>}
    <div className="row-actions">
      <button className="secondary" aria-disabled={pending || !name.trim()} onClick={() => {
        if (pending || !name.trim()) return;
        onReview({ action: album ? "rename-album" : "rename-tag", target,
          changes: album ? { name: name.trim(), description: description.trim() } : { label: name.trim(), color: color || null } }, { targetLabel });
      }}>Review rename</button>
      <button className="secondary danger" aria-disabled={pending} onClick={() => {
        if (!pending) onReview({ action: album ? "delete-album" : "delete-tag", target }, { targetLabel });
      }}>Review removal</button>
    </div>
    {!album && <>
      <label>Merge into tag<select value={destination} onChange={(event) => {
        setDestination(event.target.value); setChosenTag(choices.find((tag) => tag.id === event.target.value));
      }}>
        <option value="">Choose a different tag</option>
        {choices.filter((tag) => tag.id !== item.id).map((tag) => <option key={tag.id} value={tag.id}>{tag.label}</option>)}
      </select></label>
      <button className="secondary" aria-disabled={pending || !destination} onClick={() => {
        const selected = choices.find((tag) => tag.id === destination);
        if (!pending && selected) onReview({ action: "merge-tags", target, destination: { id: selected.id, version: selected.version } }, { targetLabel, destinationLabel: selected.label });
      }}>Review tag merge</button>
    </>}
    <p>Removing an album or tag removes its organization links. Media files stay intact.</p>
  </section>;
}

export function MediaOrganizationManager({ organization }: { organization: ReturnType<typeof useMediaOrganization> }) {
  const { creation } = organization;
  const [albumName, setAlbumName] = useState(creation.request?.kind === "albums" ? creation.request.name : "");
  const [tagName, setTagName] = useState(creation.request?.kind === "tags" ? creation.request.name : "");
  const [selected, setSelected] = useState<MediaAlbum | MediaTag | null>(null);
  const busy = creation.pending || organization.pending;
  const held = creation.request;
  const albumChoices = selected && "name" in selected && !organization.albums.some((item) => item.id === selected.id)
    ? [...organization.albums, selected] : organization.albums;
  const tagChoices = selected && "label" in selected && !organization.tags.some((item) => item.id === selected.id)
    ? [...organization.tags, selected] : organization.tags;
  const create = async (album: boolean) => {
    const name = (album ? albumName : tagName).trim();
    if (busy || !name) return;
    if (await creation.create(album ? "albums" : "tags", name)) {
      if (album) setAlbumName("");
      else setTagName("");
    }
  };
  return <AccessibleDialog title="Albums and tags" eyebrow="Media Library" closeLabel="Close albums and tags"
    onClose={() => { if (!busy) organization.setManage(false); }}>
    <p>Group media without copying their files. Existing chats and References keep their media.</p>
    <MediaOrganizationCatalogBrowser organization={organization} busy={busy} />
    <ErrorCallout message={organization.catalogFailed ? "Albums and tags could not be loaded. Refresh their choices and try again." : undefined} />
    <ErrorCallout message={organization.reviewError ?? undefined} />
    <label>New album name<input value={albumName} maxLength={200} readOnly={!!held} onChange={(event) => setAlbumName(event.target.value)} /></label>
    <button className="secondary" aria-disabled={busy || !albumName.trim() || (!!held && held.kind !== "albums")}
      onClick={() => void create(true)}>{held?.kind === "albums" ? "Try creating this album again" : "Create album"}</button>
    <label>New tag label<input value={tagName} maxLength={200} readOnly={!!held} onChange={(event) => setTagName(event.target.value)} /></label>
    <button className="secondary" aria-disabled={busy || !tagName.trim() || (!!held && held.kind !== "tags")}
      onClick={() => void create(false)}>{held?.kind === "tags" ? "Try creating this tag again" : "Create tag"}</button>
    <ErrorCallout message={creation.failed ? held
      ? "Creation could not be confirmed. Try again with the same name; the original request is kept when you close this window."
      : "Creation was refused. Edit the name and try again." : undefined} />
    <label>Edit album or tag<select value={selected?.id ?? ""} onChange={(event) => {
      setSelected([...albumChoices, ...tagChoices].find((item) => item.id === event.target.value) ?? null);
    }}>
      <option value="">Choose an album or tag</option>
      <optgroup label="Albums">{albumChoices.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</optgroup>
      <optgroup label="Tags">{tagChoices.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</optgroup>
    </select></label>
    {selected && <OrganizationEditor key={selected.id} item={selected} tags={organization.tags} pending={busy}
      onReview={(command, labels) => void organization.preview(command, undefined, labels)} />}
  </AccessibleDialog>;
}
