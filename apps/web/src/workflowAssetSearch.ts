const PATH_SEPARATOR = /[\\/]/;

/** A workflow names files, not searches: turn one into a usable query. */
export function searchTermFor(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, "").split(PATH_SEPARATOR).pop() ?? filename;
  return base.replace(/[_-]+/g, " ").trim();
}

/** The catalog role to search and preflight a workflow's asset under.
 *
 * A LoRA is an image-role asset whose lora-ness travels in `auxiliary_kind`,
 * not a role of its own. `role` is `chat`, `image` or `video` and never
 * anything else, so returning "lora" here made the server refuse every LoRA
 * with `Input should be 'chat', 'image' or 'video'` - which surfaced as a
 * button that did nothing, because a workflow's LoRAs could never be selected
 * and the installer requires a selection for *every* missing asset before it
 * will download any of them. One unselectable LoRA therefore blocked the whole
 * workflow, not just itself.
 *
 * The model library already had this right: it computes an auxiliary kind and
 * then installs under "image". This is that same rule, in the one place that
 * disagreed with it.
 */
export function catalogRoleFor(): string {
  return "image";
}
