import type { api } from "../api";

type LibraryFixtures = Pick<typeof api, "models" | "profiles">;

export async function modelPages(fixtures: LibraryFixtures, options: Parameters<typeof api.modelsPage>[0]) {
  let rows = await fixtures.models();
  if (options.chatCapability) {
    const profiles = await fixtures.profiles();
    rows = rows.filter((model) => {
      const modalities = profiles.find((profile) => profile.model_install_id === model.id)?.input_modalities ?? [];
      return model.role === "chat" && model.readiness === "ready" && (options.chatCapability === "vision"
        ? modalities.includes("image") : modalities.includes("text") && !modalities.includes("image"));
    });
  }
  const offset = options.offset ?? 0;
  return rows.slice(offset, offset + options.limit);
}

export async function catalogMatches(fixtures: LibraryFixtures, options: Parameters<typeof api.catalogInstallMatches>[0]) {
  const rows = (await fixtures.models()).filter((model) => model.role === options.role && model.active);
  return {
    remote_ids: rows.flatMap((model) => [model.manifest_json.remote_id, model.manifest_json.source_remote_id])
      .filter((id): id is string => typeof id === "string" && options.remoteIds.includes(id)),
    workflow_template_ids: rows.map((model) => model.manifest_json.workflow_template_id)
      .filter((id): id is string => typeof id === "string" && options.workflowTemplateIds.includes(id)),
  };
}


export async function presetPages(fixtures: Pick<typeof api, "presets">, options: Parameters<typeof api.presetsPage>[0]) {
  const rows = (await fixtures.presets()).filter((preset) => (!options.role || preset.role === options.role)
    && (!options.presetIds || options.presetIds.includes(preset.id))
    && (!options.defaultsOnly || preset.is_default)
    && (!options.search || preset.name.toLowerCase().includes(options.search.toLowerCase())));
  const offset = options.offset ?? 0;
  return rows.slice(offset, offset + options.limit);
}

export async function profilePages(fixtures: Pick<typeof api, "profiles">, options: Parameters<typeof api.profilesPage>[0]) {
  const rows = (await fixtures.profiles()).filter((profile) => (!options.role || profile.role === options.role)
    && (!options.engine || profile.engine === options.engine)
    && (!options.installIds || options.installIds.includes(profile.model_install_id ?? ""))
    && (!options.profileIds || options.profileIds.includes(profile.id))
    && (!options.defaultsOnly || profile.is_default)
    && (!options.inputModality || (profile.input_modalities ?? ["text"]).includes(options.inputModality))
    && (!options.search || profile.name.toLowerCase().includes(options.search.toLowerCase())));
  const offset = options.offset ?? 0;
  return rows.slice(offset, offset + options.limit);
}
