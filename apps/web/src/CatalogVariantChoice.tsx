import type { CatalogFileVariant } from "./types";

/** Formats a size the way the rest of the catalog does, without a dependency. */
function gigabytes(bytes: number | null): string {
  if (!bytes || bytes <= 0) return "";
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

/** The choice behind a filename a version publishes more than once.
 *
 * One CivitAI version can ship the same safetensors name five times at
 * different precisions. Preflight refuses to guess, which is right, and used
 * to say "choose an exact variant" while offering nothing to choose from -
 * an instruction that could not be followed.
 *
 * The rows are the server's: an immutable file id, a name, a size, and a
 * precision when the provider states one. Nothing here carries a hash or a
 * URL, because picking a variant is naming a choice, not asserting a fact.
 */
export function CatalogVariantChoice({
  filename,
  variants,
  busy = false,
  onChoose,
}: {
  filename: string;
  variants: CatalogFileVariant[];
  busy?: boolean;
  onChoose: (sourceFileId: string) => void;
}) {
  if (variants.length < 2) return null;
  return (
    <div className="catalog-variant-choice">
      <p>
        This version publishes <strong>{filename}</strong> {variants.length} times. Choose the
        one the workflow needs.
      </p>
      <ul>
        {variants.map((variant) => (
          <li key={variant.source_file_id}>
            <button
              className="secondary compact-button"
              disabled={busy}
              onClick={() => onChoose(variant.source_file_id)}
            >
              {variant.precision ?? "Unlabelled"}
              {gigabytes(variant.size_bytes) ? ` · ${gigabytes(variant.size_bytes)}` : ""}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
