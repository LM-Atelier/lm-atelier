import { app } from "../../scripts/app.js";

const PROTOCOL_VERSION = 1;
const BRIDGE_SOURCE = "lm-atelier-workflow-editor";
const PARENT_SOURCE = "lm-atelier";
const MAX_GRAPH_BYTES = 1024 * 1024;
const NONCE_PATTERN = /^[A-Za-z0-9_.-]{1,200}$/;
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

let editorPort = null;
let editorNonce = null;
let saving = false;

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function graphBytes(graph) {
  if (!isRecord(graph)) {
    return null;
  }
  try {
    return new TextEncoder().encode(JSON.stringify(graph)).byteLength;
  } catch {
    return null;
  }
}

function validGraph(graph) {
  const size = graphBytes(graph);
  return size !== null && size <= MAX_GRAPH_BYTES;
}

function isLoopbackOrigin(value) {
  try {
    const source = new URL(value);
    return (
      ["http:", "https:"].includes(source.protocol) &&
      LOOPBACK_HOSTS.has(source.hostname) &&
      source.origin === value
    );
  } catch {
    return false;
  }
}

function postPort(type, details = {}) {
  editorPort?.postMessage({
    source: BRIDGE_SOURCE,
    protocol: PROTOCOL_VERSION,
    type,
    ...details,
  });
}

function closePort() {
  const port = editorPort;
  editorPort = null;
  editorNonce = null;
  saving = false;
  if (port) {
    port.onmessage = null;
    port.onmessageerror = null;
    port.close();
  }
}

function refuse(code) {
  postPort("error", { code });
  closePort();
}

async function receivePortMessage(event) {
  const message = event.data;
  if (
    !isRecord(message) ||
    message.source !== PARENT_SOURCE ||
    message.protocol !== PROTOCOL_VERSION
  ) {
    refuse("invalid-message");
    return;
  }
  if (message.type === "cancel") {
    if (editorNonce && message.nonce !== editorNonce) {
      refuse("nonce-mismatch");
      return;
    }
    closePort();
    return;
  }
  if (
    message.type !== "load" ||
    editorNonce !== null ||
    typeof message.nonce !== "string" ||
    !NONCE_PATTERN.test(message.nonce) ||
    !validGraph(message.graph)
  ) {
    refuse("invalid-load");
    return;
  }
  try {
    await app.loadGraphData(message.graph, true, true);
  } catch {
    refuse("graph-load-failed");
    return;
  }
  editorNonce = message.nonce;
  postPort("loaded", { nonce: editorNonce });
}

function acceptParent(event) {
  const message = event.data;
  if (
    event.source !== window.opener ||
    !isLoopbackOrigin(event.origin) ||
    editorPort !== null ||
    !isRecord(message) ||
    message.source !== PARENT_SOURCE ||
    message.protocol !== PROTOCOL_VERSION ||
    message.type !== "connect" ||
    event.ports.length !== 1
  ) {
    return;
  }
  editorPort = event.ports[0];
  editorPort.onmessage = receivePortMessage;
  editorPort.onmessageerror = () => closePort();
  editorPort.start();
  postPort("connected");
}

async function saveToLmAtelier() {
  if (!editorPort || !editorNonce || saving) {
    return;
  }
  saving = true;
  try {
    const result = await app.graphToPrompt();
    const graph = result?.workflow;
    if (!validGraph(graph)) {
      refuse("invalid-save");
      return;
    }
    postPort("save", { nonce: editorNonce, graph });
    closePort();
  } catch {
    refuse("graph-save-failed");
  } finally {
    saving = false;
  }
}

if (window.opener) {
  window.addEventListener("message", acceptParent);
  window.opener.postMessage(
    {
      source: BRIDGE_SOURCE,
      protocol: PROTOCOL_VERSION,
      type: "ready",
    },
    "*",
  );
}

app.registerExtension({
  name: "LMAtelier.WorkflowEditorBridge",
  actionBarButtons: [
    {
      icon: "icon-[lucide--save]",
      label: "Save to LM Atelier",
      tooltip: "Return this workflow to LM Atelier",
      onClick: saveToLmAtelier,
    },
  ],
});
