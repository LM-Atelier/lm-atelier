// The machinery two isolated certification runs share.
//
// An isolated run is one that cannot use the ordinary browser run's assumptions:
// it needs its own loopback ports, its own data directory, its own product
// process, and a guarantee that everything it started is dead before it returns.
// That is the same work whatever the run is certifying, and it was written once
// for the browser-protocol run. A second run needs it unchanged rather than
// copied, because the parts worth getting wrong are the ones a copy would drift
// on: proving a listener is the run's own process, and killing a tree on
// Windows where a signal does not reach children.
//
// What stays in each runner is what differs - which services exist, how the
// product is configured, and which specs are the certification.

import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { access, mkdtemp, realpath, rm } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

export const TEMP_PREFIX = "lm-atelier-isolated-e2e-";
export const repositoryRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
export const apiRoot = path.join(repositoryRoot, "services", "api");

export function childExit(child) {
  return new Promise((resolve) => {
    if (child.exitCode !== null || child.signalCode !== null) {
      resolve({ code: child.exitCode, signal: child.signalCode });
      return;
    }
    child.once("error", (error) => {
      resolve({ code: null, signal: null, error: error.message });
    });
    child.once("exit", (code, signal) => resolve({ code, signal }));
  });
}

export async function reserveLoopbackPort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("Could not reserve a loopback port"));
        return;
      }
      server.close((error) => {
        if (error) reject(error);
        else resolve(address.port);
      });
    });
  });
}

async function firstExistingPath(candidates) {
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next explicit interpreter path.
    }
  }
  return undefined;
}

export async function pythonExecutable() {
  const environmentPython = process.platform === "win32"
    ? path.join(repositoryRoot, ".venv", "Scripts", "python.exe")
    : path.join(repositoryRoot, ".venv", "bin", "python");
  const projectPython = process.platform === "win32"
    ? path.join(apiRoot, ".venv", "Scripts", "python.exe")
    : path.join(apiRoot, ".venv", "bin", "python");
  const discovered = await firstExistingPath([environmentPython, projectPython]);
  return discovered ?? (process.platform === "win32" ? "python.exe" : "python3");
}

export function readinessToken() {
  return randomBytes(24).toString("base64url");
}

// A successful response is not enough: something else on the machine can be
// answering on that port. The run mints a secret and requires it back.
export async function waitForReady(url, expectedToken, output, exit, label) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const state = await Promise.race([
      exit.then((result) => ({ exited: true, result })),
      new Promise((resolve) => setTimeout(() => resolve({ exited: false }), 150)),
    ]);
    if (state.exited) {
      throw new Error(
        `${label} exited before becoming ready (${JSON.stringify(state.result)}).\n`
        + output.join(""),
      );
    }
    try {
      const response = await fetch(url);
      if (response.ok) {
        const body = await response.json();
        if (body?.token === expectedToken) return;
      }
    } catch {
      // The loopback listener is still starting or is not this run's fixture.
    }
  }
  throw new Error(`${label} did not prove readiness within 30 seconds.\n${output.join("")}`);
}

export function collectOutput(stream, output) {
  stream?.setEncoding("utf8");
  stream?.on("data", (chunk) => {
    output.push(chunk);
    if (output.length > 200) output.shift();
  });
}

async function taskkillOwnedTree(pid) {
  const killer = spawn(
    "taskkill.exe",
    ["/PID", String(pid), "/T", "/F"],
    { windowsHide: true, stdio: "ignore" },
  );
  await childExit(killer);
}

function signalOwnedTree(child, signal) {
  if (!child.pid) return;
  if (process.platform === "win32") {
    child.kill(signal);
    return;
  }
  try {
    process.kill(-child.pid, signal);
  } catch {
    child.kill(signal);
  }
}

/**
 * Stop a child, and everything it started.
 *
 * `tree` is not a tidiness option. A product that starts its own media engine
 * leaves that engine ALIVE when the product is asked to stop: on Windows a
 * termination reaches one process, never its descendants, so the engine outlives
 * the run, keeps the run's temporary directory open, and the cleanup then fails
 * with EBUSY - which is how this was found. Anything whose children are not
 * themselves owned here has to be stopped as a tree.
 */
export async function terminate(child, cancellation = false, tree = false) {
  if (!child || child.exitCode !== null || child.signalCode !== null || !child.pid) return;
  const exit = childExit(child);
  if (tree && process.platform === "win32") {
    await taskkillOwnedTree(child.pid);
    await Promise.race([exit, new Promise((resolve) => setTimeout(resolve, 5_000))]);
    return;
  }
  if (cancellation && process.platform === "win32") {
    await taskkillOwnedTree(child.pid);
    await Promise.race([exit, new Promise((resolve) => setTimeout(resolve, 5_000))]);
    return;
  }
  try {
    signalOwnedTree(child, "SIGTERM");
  } catch {
    return;
  }
  const stopped = await Promise.race([
    exit.then(() => true),
    new Promise((resolve) => setTimeout(() => resolve(false), 5_000)),
  ]);
  if (stopped) return;
  if (process.platform === "win32") {
    await taskkillOwnedTree(child.pid);
  } else {
    try {
      process.kill(-child.pid, "SIGKILL");
    } catch {
      child.kill("SIGKILL");
    }
  }
  await exit;
}

export function spawnUvicorn(python, module, port, environment) {
  return spawn(
    python,
    [
      "-m",
      "uvicorn",
      module,
      "--app-dir",
      repositoryRoot,
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
    ],
    {
      cwd: repositoryRoot,
      detached: process.platform !== "win32",
      env: environment,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
}

/**
 * Everything a run started, and the promise that it is all gone.
 *
 * Ownership is the point. A run that merely remembered its own children could
 * still leave one alive on an unexpected throw, and on Windows a signal to a
 * parent reaches nothing below it - so the tree is killed explicitly, and the
 * temporary root is only removed after a check that it is the one this run made.
 */
export class OwnedProcesses {
  constructor() {
    this.children = [];
    this.requestedSignal = undefined;
    this.cancellationCleanup = undefined;
  }

  /** @param tree stop this child's whole process tree, not just the process. */
  own(child, { tree = false } = {}) {
    this.children.push({ child, tree });
    return child;
  }

  installSignalHandlers() {
    for (const signal of ["SIGINT", "SIGTERM"]) {
      process.once(signal, () => {
        this.requestedSignal = signal;
        this.cancellationCleanup ??= this.terminateAll(true);
      });
    }
  }

  assertNotCancelled(what) {
    if (!this.requestedSignal) return;
    throw new Error(`${what} cancelled by ${this.requestedSignal}`);
  }

  async terminateAll(cancellation = false) {
    // Newest first: a test process holding the product open should go before
    // the product it is talking to.
    await Promise.allSettled(
      [...this.children].reverse().map(({ child, tree }) => terminate(child, cancellation, tree)),
    );
  }
}

export async function makeTemporaryRoot() {
  const root = await mkdtemp(path.join(tmpdir(), TEMP_PREFIX));
  await assertTemporaryRoot(root);
  return root;
}

export async function assertTemporaryRoot(root) {
  const resolvedParent = await realpath(path.dirname(root));
  const resolvedTemporaryDirectory = await realpath(tmpdir());
  if (
    resolvedParent !== resolvedTemporaryDirectory
    || !path.basename(root).startsWith(TEMP_PREFIX)
  ) {
    throw new Error(`Refusing to clean unexpected E2E directory: ${root}`);
  }
}

export async function removeTemporaryRoot(root) {
  await assertTemporaryRoot(root);
  await rm(root, { recursive: true, force: true });
}

export function sharedEnvironment(extra) {
  const pythonPath = [repositoryRoot, apiRoot, process.env.PYTHONPATH]
    .filter(Boolean)
    .join(path.delimiter);
  return {
    ...process.env,
    HTTP_PROXY: "http://127.0.0.1:9",
    HTTPS_PROXY: "http://127.0.0.1:9",
    NO_PROXY: "127.0.0.1,localhost",
    PYTHONPATH: pythonPath,
    PYTHONUNBUFFERED: "1",
    ...extra,
  };
}
