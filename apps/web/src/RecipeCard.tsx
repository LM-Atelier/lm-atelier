import { Bot, Film, Gauge, HardDrive, Image as ImageIcon } from "lucide-react";
import { formatBytes } from "./format";
import type { ReferenceRecipe } from "./types";

function recipeOperationLabel(operation: string): string {
  return operation
    .split("_")
    .map((part) => part === "to" ? "→" : part)
    .join(" ")
    .replace(/^\w/, (character) => character.toUpperCase());
}
export function RecipeCard({ recipe, pending, onInstall }: { recipe: ReferenceRecipe; pending: boolean; onInstall: () => void }) {
  const memory = recipe.hardware.minimum_vram_gb
    ? `${recipe.hardware.minimum_vram_gb} GB+ VRAM`
    : `${recipe.hardware.minimum_ram_gb} GB+ RAM`;
  return (
    <article className="recipe-card">
      <header>
        <div className="model-icon">{recipe.role === "video" ? <Film /> : recipe.role === "image" ? <ImageIcon /> : <Bot />}</div>
        <div><small>{recipe.role} · recipe v{recipe.version}</small><h3>{recipe.name}</h3></div>
      </header>
      <p>{recipe.summary}</p>
      <div className="recipe-badges"><span className={`badge ${recipe.certified ? "likely" : ""}`}>{recipe.certified ? "Certified" : "Reference candidate"}</span>{recipe.operations.map((operation) => <span className="badge" key={operation}>{recipeOperationLabel(operation)}</span>)}<span className="badge">{recipe.license_id}</span><span className="badge">{recipe.node_policy || recipe.engine}</span></div>
      <div className="recipe-meta"><span><HardDrive size={14} />{formatBytes(recipe.total_size_bytes)}</span><span><Gauge size={14} />{memory}</span></div>
      <small>{recipe.hardware.guidance}</small>
      <button className="primary" onClick={onInstall} disabled={pending}>{pending ? "Queued" : "Install recipe"}</button>
    </article>
  );
}
