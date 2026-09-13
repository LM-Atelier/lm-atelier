/** The Settings destinations, and which one a stored choice means.
 *
 * Its own file because the identifiers outlive any one render: they are what a
 * remembered choice, a deep link and a test all name. Keeping them beside the
 * component would make every consumer import a component to ask a question that
 * needs no rendered tree.
 *
 * GENERAL IS NOT HERE YET, and its absence is deliberate rather than an
 * oversight. It is a destination for behaviour the product does not have -
 * startup destination, reopen behaviour, send-key behaviour - so shipping it now
 * would add an empty page. It arrives with its content. The ids below do not
 * shift when it does: a destination is identified by its own id, never by
 * position.
 *
 * Settings opens on the first entry, so the order also decides where a person
 * lands. General takes the first place when it exists.
 */

export interface SettingsDestination {
  id: string;
  label: string;
  /** What a person can expect to find, shown under the heading. */
  summary: string;
}

export const SETTINGS_DESTINATIONS: readonly SettingsDestination[] = [
  {
    id: "appearance",
    label: "Appearance",
    summary: "Light or dark, and which theme.",
  },
  {
    id: "models-and-generation",
    label: "Models & generation",
    summary: "Model profiles and generation presets.",
  },
  {
    id: "model-sources",
    label: "Model sources",
    summary: "Where models are downloaded from, and the credentials that allow it.",
  },
  {
    id: "data-and-backups",
    label: "Data & backups",
    summary: "Where disk space goes, clearing media nothing uses, and recovery backups.",
  },
  {
    id: "advanced",
    label: "Advanced",
    summary: "Web search, engines, this machine, and the managed workers that run them.",
  },
  {
    id: "about-and-support",
    label: "About & support",
    summary: "Version, where things are stored, and where to get help.",
  },
];

export const DEFAULT_SETTINGS_DESTINATION = SETTINGS_DESTINATIONS[0].id;

/** The destination a remembered or supplied id means, or the default.
 *
 * An id that no longer exists resolves to the default rather than to nothing,
 * because a person returning after a destination was renamed should land
 * somewhere usable rather than on a blank page.
 */
export function settingsDestinationFor(id: unknown): SettingsDestination {
  const match = SETTINGS_DESTINATIONS.find((destination) => destination.id === id);
  return match ?? SETTINGS_DESTINATIONS[0];
}
