import { useEffect, useState } from "react";

/** The decoded picture for one artifact, and what went wrong if there is none.
 *
 * Kept as one state deliberately. Held apart, a bitmap could outlive the
 * artifact it was decoded for; a decode finishing after the selection moved
 * leaked its memory; and every failure - a 404, a body that is not an image,
 * a dropped connection - arrived as "no bitmap yet" and rendered as loading,
 * forever and without saying so.
 */
export function useStudioImage(artifactId: string | null) {
  const [loaded, setLoaded] = useState<{
    artifactId: string | null;
    bitmap: ImageBitmap | null;
    error: string | null;
  }>({ artifactId: null, bitmap: null, error: null });
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    let live = true;
    const abort = new AbortController();
    const replace = (next: { bitmap: ImageBitmap | null; error: string | null }) =>
      setLoaded((previous) => {
        // The picture being replaced holds decoded memory until it is closed,
        // and nothing else will close it.
        if (previous.bitmap && previous.bitmap !== next.bitmap) previous.bitmap.close();
        return { artifactId, ...next };
      });

    if (!artifactId) {
      // Async so the clear never runs synchronously inside the effect.
      void Promise.resolve().then(() => {
        if (live) replace({ bitmap: null, error: null });
      });
      return () => {
        live = false;
      };
    }

    void fetch(`/api/artifacts/${encodeURIComponent(artifactId)}/content`, {
      signal: abort.signal,
    })
      .then((response) => {
        // An error response has a body too, and it decodes to nothing. Without
        // this, a refusal arrived as an undecodable image and was reported as
        // no image at all.
        if (!response.ok) throw new Error(`This picture could not be read (${response.status}).`);
        return response.blob();
      })
      .then((blob) => createImageBitmap(blob))
      .then((decoded) => {
        // A decode that lands after the effect was replaced belongs to a
        // picture nobody is looking at, and holds its memory until closed.
        if (!live) {
          decoded.close();
          return;
        }
        replace({ bitmap: decoded, error: null });
      })
      .catch((reason: unknown) => {
        if (!live || abort.signal.aborted) return;
        replace({
          bitmap: null,
          error: reason instanceof Error ? reason.message : "This picture could not be read.",
        });
      });

    return () => {
      live = false;
      abort.abort();
    };
  }, [artifactId, reloads]);

  return {
    bitmap: loaded.artifactId === artifactId ? loaded.bitmap : null,
    error: loaded.artifactId === artifactId ? loaded.error : null,
    reload: () => setReloads((attempt) => attempt + 1),
  };
}
