import { Fragment, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { PictureRemixDialog } from "./PictureRemixDialog";
import {
  ignoredReason,
  pictureSettingsCut,
  pictureSettingsNote,
  readPictureSettings,
  settingLabel,
} from "./pictureSettings";

function PictureFileSettingsBody({
  artifactId,
  onOpenChat,
}: {
  artifactId: string;
  onOpenChat?: (chatId: string) => void;
}) {
  const [remixing, setRemixing] = useState(false);
  const settings = useQuery({
    queryKey: ["picture-file-settings", artifactId],
    queryFn: async ({ signal }) => readPictureSettings(await api.pictureSettings(artifactId, signal)),
    retry: false,
    // A stored picture never changes, so one reading holds for as long as it is shown.
    staleTime: Infinity,
  });
  if (settings.isPending) return <p role="status">Reading the picture's file…</p>;
  if (settings.isError) {
    const failure = settings.error as { code?: unknown } | null;
    return <p role="alert">{failure?.code === "generation-settings-unreadable"
      ? "The settings text stored in this picture could not be read."
      : "This picture's file could not be read."}</p>;
  }
  const note = pictureSettingsNote(settings.data);
  return <div className="generation-details-content">
    {note && <p>{note}</p>}
    {settings.data.claims.length > 0 && <dl className="generation-identity">
      {settings.data.claims.map((claim) => <Fragment key={claim.key}>
        <dt>{settingLabel(claim.key)}</dt><dd>{String(claim.value)}</dd>
      </Fragment>)}
    </dl>}
    {settings.data.ignored.length > 0 && <>
      <strong>Left out</strong>
      <ul>{settings.data.ignored.map((item, index) => <li key={index}>
        {item.name}: {ignoredReason(item.reason)}
      </li>)}</ul>
    </>}
    {pictureSettingsCut(settings.data) && <p>More was left out than is listed here.</p>}
    <p>Read from the picture's own file. Nothing named in it is fetched, run or kept.</p>
    {settings.data.claims.some((claim) => claim.key === "prompt") && (
      <button type="button" className="secondary compact-button" onClick={() => setRemixing(true)}>
        Remix these settings
      </button>
    )}
    {/* Outside the card it is opened from, so the card's own styles never reach the dialog. */}
    {remixing && createPortal(
      <PictureRemixDialog
        artifactId={artifactId}
        onClose={() => setRemixing(false)}
        onOpenChat={onOpenChat}
      />,
      document.body,
    )}
  </div>;
}

/** The settings a picture made elsewhere carries in its own file, read when opened. */
export function PictureFileSettings({
  artifactId,
  onOpenChat,
}: {
  artifactId: string;
  onOpenChat?: (chatId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  return <details className="generation-details" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Settings in the file</summary>
    {open && <PictureFileSettingsBody artifactId={artifactId} onOpenChat={onOpenChat} />}
  </details>;
}
