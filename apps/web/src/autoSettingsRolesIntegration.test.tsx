import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { AUTO_SETTINGS_ROLES_KEY } from "./autoSettingsRoles";
import { SettingsDrawer } from "./SettingsDrawer";
import type { EngineCapabilities, EngineRole } from "./types";
import { useAutoSettingsRoles } from "./useAutoSettingsRoles";

/**
 * The contracts that only hold once something is mounted.
 *
 * The pure-helper tests prove the rules in isolation; they cannot prove that a
 * reload actually restores the history, that two chats keep separate ones, or
 * that removing `key={role}` really preserves the disclosure level - that last
 * one is a claim about React's reconciliation, not about a function.
 */

const ENGINES: EngineCapabilities[] = [
  {
    id: "engine-1",
    name: "Test engine",
    roles: ["chat", "image", "video"],
    fields: [
      { key: "basic_field", label: "Basic field", type: "number", visibility: "basic" },
      { key: "adv_field", label: "Advanced field", type: "number", visibility: "advanced" },
    ],
  } as unknown as EngineCapabilities,
];

function Harness({ chats }: { chats: readonly { id: string }[] | undefined }) {
  const [roles, remember] = useAutoSettingsRoles(chats);
  const [chatId, setChatId] = useState("chat-a");
  return (
    <div>
      <span data-testid="current">{roles[chatId] ?? "none"}</span>
      <button type="button" onClick={() => setChatId("chat-a")}>use a</button>
      <button type="button" onClick={() => setChatId("chat-b")}>use b</button>
      <button type="button" onClick={() => remember(chatId, "image")}>pick image</button>
      <button type="button" onClick={() => remember(chatId, "video")}>pick video</button>
    </div>
  );
}

beforeEach(() => localStorage.clear());
afterEach(cleanup);

describe("remembered roles, mounted", () => {
  it("restores the picked role after a remount, which is what a reload is", () => {
    const chats = [{ id: "chat-a" }, { id: "chat-b" }];
    const first = render(<Harness chats={chats} />);
    fireEvent.click(screen.getByText("pick image"));
    expect(screen.getByTestId("current").textContent).toBe("image");
    first.unmount();

    render(<Harness chats={chats} />);
    expect(screen.getByTestId("current").textContent).toBe("image");
  });

  it("keeps a separate history per chat", () => {
    const chats = [{ id: "chat-a" }, { id: "chat-b" }];
    render(<Harness chats={chats} />);
    fireEvent.click(screen.getByText("pick image"));
    fireEvent.click(screen.getByText("use b"));
    expect(screen.getByTestId("current").textContent).toBe("none");
    fireEvent.click(screen.getByText("pick video"));
    fireEvent.click(screen.getByText("use a"));
    expect(screen.getByTestId("current").textContent).toBe("image");
  });

  it("prunes nothing before the chat list resolves, and clears once it resolves empty", () => {
    // The two states an earlier version collapsed. Undefined is "not known
    // yet"; an empty array is a loaded list with no chats in it.
    render(<Harness chats={undefined} />);
    fireEvent.click(screen.getByText("pick image"));
    cleanup();

    render(<Harness chats={undefined} />);
    expect(screen.getByTestId("current").textContent).toBe("image");
    cleanup();

    render(<Harness chats={[]} />);
    expect(JSON.parse(localStorage.getItem(AUTO_SETTINGS_ROLES_KEY) ?? "{}").roles).toEqual({});
  });
});

function drawer(role: EngineRole, onRole: (next: EngineRole) => void) {
  return (
    <SettingsDrawer
      open
      onClose={() => {}}
      mode="auto"
      role={role}
      onRole={onRole}
      engines={ENGINES}
      values={{}}
      onValues={() => {}}
      presets={[]}
      presetId={null}
      onPreset={() => {}}
      imageEdit={false}
      imageEditPrompt=""
    />
  );
}

describe("the two stacked pickers", () => {
  it("are separately named groups, not one control repeated", () => {
    render(drawer("chat", () => {}));
    expect(screen.getByRole("group", { name: "Settings role" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Settings detail level" })).toBeTruthy();
  });

  it("keeps the chosen detail level when the role changes", () => {
    // This is the whole point of removing key={role}. The panel derives
    // everything else from its props, so the key was resetting only the
    // reader's disclosure choice - which has nothing to do with which role is
    // being edited.
    function Switcher() {
      const [role, setRole] = useState<EngineRole>("chat");
      return drawer(role, setRole);
    }
    render(<Switcher />);

    fireEvent.click(screen.getByRole("button", { name: "advanced" }));
    expect(screen.getByRole("button", { name: "advanced" }).className).toContain("active");

    fireEvent.click(screen.getByRole("button", { name: "image" }));
    expect(screen.getByRole("button", { name: "advanced" }).className).toContain("active");
  });
});
