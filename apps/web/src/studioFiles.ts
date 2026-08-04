/** Whether a drop or a pick is something the studio can open. */
export function firstImage(files: readonly File[]): File | null {
  return files.find((file) => file.type.startsWith("image/")) ?? null;
}
