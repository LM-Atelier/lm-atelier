import { SETTINGS_DESTINATIONS } from "./settingsDestinations";

/** Moving between Settings destinations, on a rail or in a picker.
 *
 * These are PAGES, not tabs, and the difference is not cosmetic. A tab widget
 * promises arrow-key traversal within one composite control and a panel that
 * belongs to the tab; a destination list promises that each entry takes you
 * somewhere and that the current one is announced as the current page. The
 * second is what this is, so it renders a `nav` of ordinary buttons with
 * `aria-current="page"` rather than a tablist.
 *
 * THE SAME DESTINATIONS APPEAR TWICE, once as a rail and once as a select, and
 * only one is visible at a time. That is deliberate rather than duplication:
 * a rail is unusable at phone width and a select is a poor way to show five
 * choices when there is room for all of them. Both drive the same state, so
 * neither can disagree with the other, and the select carries a real label
 * rather than a placeholder option.
 */
export function SettingsNavigation({
  current,
  onSelect,
}: {
  current: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="settings-navigation">
      <nav className="settings-rail" aria-label="Settings sections">
        {SETTINGS_DESTINATIONS.map((destination) => (
          <button
            key={destination.id}
            type="button"
            className={destination.id === current ? "active" : ""}
            aria-current={destination.id === current ? "page" : undefined}
            onClick={() => onSelect(destination.id)}
          >
            {destination.label}
          </button>
        ))}
      </nav>
      <label className="settings-picker">
        <span>Settings section</span>
        <select value={current} onChange={(event) => onSelect(event.target.value)}>
          {SETTINGS_DESTINATIONS.map((destination) => (
            <option key={destination.id} value={destination.id}>
              {destination.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
