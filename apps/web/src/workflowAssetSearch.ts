const PATH_SEPARATOR = /[\\/]/;

/** A workflow names files, not searches: turn one into a usable query. */
export function searchTermFor(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, "").split(PATH_SEPARATOR).pop() ?? filename;
  return base.replace(/[_-]+/g, " ").trim();
}

export function catalogRoleFor(kind: string): string {
  return kind === "lora" ? "lora" : "image";
}
