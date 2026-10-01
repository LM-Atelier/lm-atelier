import { useState } from "react";
import { ProjectPageControls } from "./ProjectPageControls";
import { distinctProjects, useProject, useProjectPages } from "./useProjectPages";

export function ProjectPicker({ value, onChange, label = "Project", emptyLabel = "Unfiled" }: {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  emptyLabel?: string;
}) {
  const [search, setSearch] = useState("");
  const pages = useProjectPages(search);
  const selected = useProject(value);
  const projects = distinctProjects([
    ...(pages.data ?? []).filter(project => project.id !== value),
    ...(selected.isSuccess && selected.data?.id === value ? [selected.data] : []),
  ]);
  return <>
    <label>Search projects<input type="search" maxLength={500} value={search} onChange={(event) => setSearch(event.target.value)} /></label>
    <label>{label}<select value={value} disabled={pages.isPending || Boolean(pages.error)} onChange={(event) => onChange(event.target.value)}>
      <option value="">{pages.isPending ? "Loading projects…" : pages.error ? "Cannot read projects" : emptyLabel}</option>
      {value && !projects.some((project) => project.id === value) && <option value={value} disabled>
        {selected.isPending ? "Loading selected project…" : selected.isError ? "Selected project" : "Selected project unavailable"}
      </option>}
      {projects.map((project) => <option key={project.id} value={project.id}>
        {project.name}{project.archived ? " (Archived)" : ""}
      </option>)}
    </select></label>
    {selected.error && <p role="alert">{selected.error.message} <button type="button" onClick={() => void selected.refetch()}>Retry selected project</button></p>}
    <ProjectPageControls pages={pages} />
  </>;
}
