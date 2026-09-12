/** The Settings destinations, and which one a stored choice means.
 *
 * Its own file because the identifiers outlive any one render: they are what a
 * remembered choice, a deep link and a test all name. Keeping them beside the
 * component would make every consumer import a component to ask a question that
 * needs no rendered tree.
 *
 * TWO OF THE APPROVED SEVEN ARE NOT HERE YET, and their absence is deliberate
 * rather than an oversight. General and Appearance are destinations for
 * behaviour the product does not have - startup destination, reopen behaviour,
 * theme room, density, text size - so shipping them now would be five useful
 * pages and two empty ones. They arrive with their content, which is what
 * "separately reviewable changes" was asked for. The ids below do not shift
 * when they do: a destination is identified by its own id, never by position.
 */

export interface SettingsDestination {
  id: string;
  label: string;
  /** What a person can expect to find, shown under the heading. */
  summary: string;
}

export const SETTINGS_DESTINATIONS: readonly SettingsDestination[] = [
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
    summary: "Recovery backups of your library and settings.",
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
