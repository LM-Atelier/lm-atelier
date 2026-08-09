const PROTOCOL_VERSION = 2;
const BRIDGE_SOURCE = "lm-atelier-workflow-editor";
const PARENT_SOURCE = "lm-atelier";

const frame = document.getElementById("workflow-editor-frame");
const status = document.getElementById("workflow-editor-status");
const openerWindow = window.opener;
const comfyOrigin = new URL(frame.src).origin;

let bridgeReady = false;
let connected = false;

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function validEnvelope(message, source, type) {
  return (
    isRecord(message) &&
    message.source === source &&
    message.protocol === PROTOCOL_VERSION &&
    message.type === type
  );
}

function showStatus(message) {
  status.textContent = message;
  status.hidden = false;
}

function acceptBridgeReady(event) {
  if (
    bridgeReady ||
    event.source !== frame.contentWindow ||
    event.origin !== comfyOrigin ||
    !validEnvelope(event.data, BRIDGE_SOURCE, "ready") ||
    event.ports.length !== 0
  ) {
    return;
  }
  bridgeReady = true;
  status.hidden = true;
  openerWindow.postMessage(event.data, window.location.origin);
}

function acceptOpenerConnect(event) {
  if (
    !bridgeReady ||
    connected ||
    event.source !== openerWindow ||
    event.origin !== window.location.origin ||
    !validEnvelope(event.data, PARENT_SOURCE, "connect") ||
    event.ports.length !== 1
  ) {
    return;
  }
  connected = true;
  frame.contentWindow.postMessage(event.data, comfyOrigin, [event.ports[0]]);
}

if (!openerWindow) {
  showStatus("Open the workflow editor from LM Atelier.");
} else {
  window.addEventListener("message", (event) => {
    if (event.source === frame.contentWindow) {
      acceptBridgeReady(event);
      return;
    }
    if (event.source === openerWindow) {
      acceptOpenerConnect(event);
    }
  });
}
