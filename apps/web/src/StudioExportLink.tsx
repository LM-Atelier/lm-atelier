import { Download } from "lucide-react";
import { useState } from "react";
import { artifactSource, exportSource, type ExportFormat } from "./messageMedia";

const FORMATS: Array<{ value: ExportFormat | "stored"; label: string }> = [
  { value: "stored", label: "As stored" },
  { value: "png", label: "PNG" },
  { value: "jpeg", label: "JPEG" },
  { value: "webp", label: "WebP" },
];

/** Exporting the picture on screen, as stored or in another format.
 *
 * As stored is the file itself, byte for byte. The other formats are made from
 * it on the way out: upright, with its color profile, and for JPEG laid on white
 * where it was transparent. The picture in the studio does not change either way.
 */
export function StudioExportLink({ artifactId }: { artifactId: string }) {
  const [format, setFormat] = useState<ExportFormat | "stored">("stored");
  const href = format === "stored" ? artifactSource(artifactId) : exportSource(artifactId, format);
  return (
    <span className="studio-export">
      <select
        aria-label="Export format"
        value={format}
        onChange={(event) => setFormat(event.target.value as ExportFormat | "stored")}
      >
        {FORMATS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <a
        className="secondary compact-button"
        href={href ?? undefined}
        download
        title={format === "jpeg" ? "JPEG has no transparency: transparent parts come out white." : undefined}
      >
        <Download size={14} aria-hidden="true" /> Export
      </a>
    </span>
  );
}
