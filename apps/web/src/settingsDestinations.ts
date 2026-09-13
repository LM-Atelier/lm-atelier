/** The Settings destinations, and which one a stored choice means.
 *
 * Its own file because the identifiers outlive any one render: they are what a
 * remembered choice, a deep link and a test all name. Keeping them beside the
 * component would make every consumer import a component to ask a question that
 * needs no rendered tree.
 *
 * General is first, so Settings opens on it. It holds everyday behaviour that is
 * not about appearance, and grows as that behaviour arrives: a destination is
 * identified by its own id, never by position, so adding to it moves nothing.
 */

export interface SettingsDestination {
  id: string;
  label: string;
  /** What a person can expect to find, shown under the heading. */
  summary: string;
}

export const SETTINGS_DESTINATIONS: readonly SettingsDestination[] = [
  {
    id: "general",
    label: "General",
    summary: "Which key sends a message, and whether finished work is announced.",
  },
  {
    id: "appearance",
    label: "Appearance",
    summary: "Light or dark, theme, chat width, motion, the sidebar and the clock.",
  },
  {
    id: "models-and-generation",
    label: "Models & generation",
    summary: "Model profiles, generation presets, and how much detail their editors open with.",
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
