import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { WorkflowReadPageControls } from "./WorkflowReadPageControls";
import type { WorkflowFamilyBrowseState } from "./useWorkflowFamilyChoices";
import "./WorkflowChoiceDropdown.css";

export type WorkflowDropdownOption = {
  value: string; label: string; disabled?: boolean; searchResult?: boolean;
};

export function WorkflowChoiceDropdown({ id, label, browseLabel, value, options, browse, saving, unavailableText, onChange }: {
  id: string; label: string; value: string; options: WorkflowDropdownOption[];
  browseLabel: string;
  unavailableText?: string;
  browse: WorkflowFamilyBrowseState; saving: boolean; onChange: (value: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const blocked = saving || unavailableText !== undefined;
  const expanded = open && unavailableText === undefined;
  const listId = `${id}-choices`;
  const selectedLabel = options.find(option => option.value === value)?.label ?? "Selected workflow";
  const query = browse.search.trim().toLowerCase();
  const visible = options.filter(option => !query || option.searchResult || option.label.toLowerCase().includes(query));
  const enabled = visible.filter(option => !option.disabled);
  const activeIndex = visible.findIndex(option => option.value === active && !option.disabled);
  const shown = unavailableText ?? (open ? browse.search : selectedLabel);
  useEffect(() => {
    if (open && activeIndex >= 0) document.getElementById(`${listId}-${activeIndex}`)?.scrollIntoView?.({ block: "nearest" });
  }, [open, activeIndex, listId]);
  useEffect(() => {
    // After a choice or Escape the list closes with focus kept in the box,
    // which then shows the chosen workflow's name. Selected, that name is
    // replaced by the next keystroke, which starts a new search instead of
    // being added to the end of it.
    if (!open && document.activeElement === input.current) input.current?.select();
  }, [open, shown]);
  const close = () => { setOpen(false); setActive(null); browse.setSearch(""); };
  const choose = (next: string) => {
    if (blocked || !enabled.some(option => option.value === next)) return;
    onChange(next);
    close();
    input.current?.focus();
  };
  return <div className="workflow-choice-dropdown" onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget)) close();
  }}>
    <div className="workflow-choice-input">
      <input ref={input} id={id} role="combobox" aria-autocomplete="list" aria-expanded={expanded}
        aria-controls={expanded ? listId : undefined} aria-activedescendant={expanded && activeIndex >= 0 ? `${listId}-${activeIndex}` : undefined}
        aria-disabled={blocked} readOnly={blocked} autoComplete="off" maxLength={500}
        value={shown} placeholder={open ? "Type to search…" : selectedLabel}
        onFocus={event => event.currentTarget.select()}
        onClick={() => { if (!blocked) setOpen(true); }}
        onChange={event => { if (!blocked) { browse.setSearch(event.target.value); setOpen(true); setActive(null); } }}
        onKeyDown={event => {
          if (event.key === "Escape" && open) { event.preventDefault(); event.stopPropagation(); close(); }
          else if (!blocked && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
            event.preventDefault();
            setOpen(true);
            const index = enabled.findIndex(option => option.value === active);
            const next = index < 0 ? (event.key === "ArrowDown" ? 0 : enabled.length - 1)
              : (index + (event.key === "ArrowDown" ? 1 : -1) + enabled.length) % enabled.length;
            setActive(enabled[next]?.value ?? null);
          } else if (event.key === "Enter" && open) {
            event.preventDefault();
            if (active !== null) choose(active);
          }
        }} />
      <button type="button" className="icon-button" aria-label={`Show ${label.toLowerCase()} choices`}
        aria-expanded={expanded} aria-disabled={blocked} tabIndex={-1}
        onClick={() => { if (!blocked) { if (open) close(); else setOpen(true); input.current?.focus(); } }}>
        <ChevronDown aria-hidden="true" size={15} />
      </button>
    </div>
    {expanded && <div className="workflow-choice-popup">
      <div id={listId} role="listbox" aria-label={`${label} choices`} className="workflow-choice-list">
        {visible.map((option, index) => <button key={option.value} id={`${listId}-${index}`} type="button"
          role="option" tabIndex={-1} aria-selected={option.value === value} aria-disabled={Boolean(option.disabled || saving)}
          className={`workflow-choice-option${option.value === active ? " active" : ""}`}
          onMouseDown={event => event.preventDefault()} onClick={() => choose(option.value)}>
          {option.label}
        </button>)}
      </div>
      {!visible.length && !browse.pages.isPending && !browse.pages.error && <p role="status">No matching workflows.</p>}
      <WorkflowReadPageControls pages={browse.pages} label={browseLabel}
        onEscape={() => { close(); input.current?.focus(); }} />
    </div>}
  </div>;
}
