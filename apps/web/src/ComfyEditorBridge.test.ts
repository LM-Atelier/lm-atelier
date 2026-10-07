import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it, vi } from "vitest";

const script = readFileSync(resolve(
  "../../services/api/local_lm/comfy_editor_bridge_assets/js/lm_atelier_workflow_editor.js",
), "utf8").replace(/^import .*;\r?\n/gm, "");

type Port = {
  start: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  postMessage: ReturnType<typeof vi.fn>;
  onmessage: ((event: { data: unknown }) => Promise<void>) | null;
  onmessageerror: (() => void) | null;
};

function openBridge() {
  const coordinator = { postMessage: vi.fn() };
  const handlers = new Map<string, (event: unknown) => void>();
  const host = {
    parent: coordinator,
    opener: null,
    addEventListener: (name: string, handler: (event: unknown) => void) => handlers.set(name, handler),
  };
  let extension: { setup?: () => void } | undefined;
  const app = {
    registerExtension: (value: typeof extension) => { extension = value; },
    loadGraphData: vi.fn().mockResolvedValue(undefined),
    graphToPrompt: vi.fn(),
  };
  new Function("app", "window", "COORDINATOR_ORIGINS", script)(
    app, host, ["http://127.0.0.1:12340"],
  );
  const port: Port = { start: vi.fn(), close: vi.fn(), postMessage: vi.fn(), onmessage: null, onmessageerror: null };
  const connect = () => handlers.get("message")?.({
    source: coordinator,
    origin: "http://127.0.0.1:12340",
    data: { source: "lm-atelier", protocol: 2, type: "connect" },
    ports: [port],
  });
  return { coordinator, app, port, connect, setup: () => extension?.setup?.() };
}

it("announces readiness only after frontend setup and only once", () => {
  const bridge = openBridge();
  expect(bridge.coordinator.postMessage).not.toHaveBeenCalled();
  bridge.setup();
  expect(bridge.coordinator.postMessage).toHaveBeenCalledExactlyOnceWith(
    { source: "lm-atelier-workflow-editor", protocol: 2, type: "ready" }, "*",
  );
  bridge.setup();
  expect(bridge.coordinator.postMessage).toHaveBeenCalledTimes(1);
});

it("ignores early connections and loads through one port after setup", async () => {
  const bridge = openBridge();
  bridge.connect();
  expect(bridge.port.start).not.toHaveBeenCalled();
  bridge.setup();
  bridge.connect();
  expect(bridge.port.start).toHaveBeenCalledOnce();
  await bridge.port.onmessage?.({ data: {
    source: "lm-atelier", protocol: 2, type: "load", nonce: "neutral-editor-session", graph: { nodes: [] },
  } });
  expect(bridge.app.loadGraphData).toHaveBeenCalledExactlyOnceWith({ nodes: [] }, true, true);
  expect(bridge.port.postMessage).toHaveBeenLastCalledWith({
    source: "lm-atelier-workflow-editor", protocol: 2, type: "loaded", nonce: "neutral-editor-session",
  });
  bridge.connect();
  expect(bridge.port.start).toHaveBeenCalledOnce();
});
