export function catalogUnavailableMessage(source: string) {
  const provider = source === "civitai" ? "CivitAI" : "Hugging Face";
  return `Showing saved results while ${provider} is unavailable.`;
}
