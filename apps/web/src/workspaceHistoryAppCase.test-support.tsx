import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { expect } from "vitest";

export async function exerciseWorkspaceHistory(renderApp: () => unknown) {
  const state = { launcher: "neutral-fixture" };
  window.history.replaceState(state, "", "/?view=settings&settings=advanced&keep=1#section");
  renderApp();
  const advanced = await screen.findByRole("region", { name: "Advanced" });
  expect(document.activeElement).toBe(document.body);
  fireEvent.click(screen.getByRole("button", { name: "Data & backups" }));
  const backups = screen.getByRole("region", { name: "Data & backups" });
  expect(document.activeElement).toBe(backups);
  expect(new URL(window.location.href).searchParams.get("settings")).toBe("data-and-backups");
  expect(window.history.state).toEqual(state);
  expect(window.location.hash).toBe("#section");
  expect(new URL(window.location.href).searchParams.get("keep")).toBe("1");
  act(() => window.history.back());
  await waitFor(() => expect(screen.getByRole("region", { name: "Advanced" })).toBe(advanced));
  await waitFor(() => expect(document.activeElement).toBe(advanced));
  act(() => window.history.forward());
  await waitFor(() => expect(screen.getByRole("region", { name: "Data & backups" })).toHaveFocus());
  fireEvent.click(screen.getByRole("button", { name: "Workflows" }));
  await waitFor(() => expect(new URL(window.location.href).searchParams.get("view")).toBe("workflows"));
  await waitFor(() => expect(document.getElementById("main-content")).toHaveFocus());
  act(() => window.history.back());
  await waitFor(() => expect(screen.getByRole("region", { name: "Data & backups" })).toHaveFocus());
  act(() => window.history.forward());
  await waitFor(() => expect(new URL(window.location.href).searchParams.get("view")).toBe("workflows"));
  await waitFor(() => expect(document.getElementById("main-content")).toHaveFocus());
}
