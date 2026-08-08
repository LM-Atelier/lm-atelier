import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { access, mkdir, mkdtemp, realpath, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const TEMP_PREFIX = "lm-atelier-workflow-editor-e2e-";
const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const apiRoot = path.join(repositoryRoot, "services", "api");
const require = createRequire(import.meta.url);
let productProcess;
let comfyProcess;
let attackerProcess;
let testProcess;
let requestedSignal;
let cancellationCleanup;

function childExit(child) {
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

async function reserveLoopbackPort() {
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

async function pythonExecutable() {
  const configured = process.env.LM_ATELIER_E2E_PYTHON?.trim();
  if (configured) return configured;
  const environmentPython = process.platform === "win32"
    ? path.join(repositoryRoot, ".venv", "Scripts", "python.exe")
    : path.join(repositoryRoot, ".venv", "bin", "python");
  const projectPython = process.platform === "win32"
    ? path.join(apiRoot, ".venv", "Scripts", "python.exe")
    : path.join(apiRoot, ".venv", "bin", "python");
  const discovered = await firstExistingPath([environmentPython, projectPython]);
  return discovered ?? (process.platform === "win32" ? "python.exe" : "python3");
}

async function waitForReady(url, expectedToken, output, exit) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const state = await Promise.race([
      exit.then((result) => ({ exited: true, result })),
      new Promise((resolve) => setTimeout(() => resolve({ exited: false }), 150)),
    ]);
    if (state.exited) {
      throw new Error(
        `Browser fixture exited before becoming ready (${JSON.stringify(state.result)}).\n`
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
  throw new Error(`Browser fixture did not prove readiness within 30 seconds.\n${output.join("")}`);
}

function collectOutput(stream, output) {
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

async function terminate(child, cancellation = false) {
  if (!child || child.exitCode !== null || child.signalCode !== null || !child.pid) return;
  const exit = childExit(child);
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

async function terminateOwnedProcesses(cancellation = false) {
  await Promise.allSettled([
    terminate(testProcess, cancellation),
    terminate(productProcess, cancellation),
    terminate(comfyProcess, cancellation),
    terminate(attackerProcess, cancellation),
  ]);
}

async function validateTemporaryRoot(root) {
  const resolvedParent = await realpath(path.dirname(root));
  const resolvedTemporaryDirectory = await realpath(tmpdir());
  if (
    resolvedParent !== resolvedTemporaryDirectory
    || !path.basename(root).startsWith(TEMP_PREFIX)
  ) {
    throw new Error(`Refusing to clean unexpected E2E directory: ${root}`);
  }
}

function installSignalHandlers() {
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.once(signal, () => {
      requestedSignal = signal;
      cancellationCleanup ??= terminateOwnedProcesses(true);
    });
  }
}

function spawnUvicorn(python, module, port, environment) {
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

function readinessToken() {
  return randomBytes(24).toString("base64url");
}

function assertNotCancelled() {
  if (!requestedSignal) return;
  throw new Error(`Browser protocol certification cancelled by ${requestedSignal}`);
}

async function main() {
  installSignalHandlers();
  const temporaryRoot = await mkdtemp(path.join(tmpdir(), TEMP_PREFIX));
  await validateTemporaryRoot(temporaryRoot);
  const dataDirectory = path.join(temporaryRoot, "data");
  const fixtureDirectory = path.join(temporaryRoot, "fixture");
  const outputDirectory = path.join(temporaryRoot, "playwright-output");
  await mkdir(fixtureDirectory, { recursive: true });

  const appPort = await reserveLoopbackPort();
  const comfyPort = await reserveLoopbackPort();
  const attackerPort = await reserveLoopbackPort();
  const baseURL = `http://127.0.0.1:${appPort}`;
  const comfyOrigin = `http://127.0.0.1:${comfyPort}`;
  const attackerOrigin = `http://127.0.0.1:${attackerPort}`;
  const readyTokens = {
    product: readinessToken(),
    comfy: readinessToken(),
    attacker: readinessToken(),
  };
  const python = await pythonExecutable();
  const pythonPath = [repositoryRoot, apiRoot, process.env.PYTHONPATH]
    .filter(Boolean)
    .join(path.delimiter);
  const sharedEnvironment = {
    ...process.env,
    LM_ATELIER_E2E_APP_ORIGIN: baseURL,
    LM_ATELIER_E2E_ATTACKER_ORIGIN: attackerOrigin,
    LM_ATELIER_E2E_FIXTURE_ROOT: fixtureDirectory,
    HTTP_PROXY: "http://127.0.0.1:9",
    HTTPS_PROXY: "http://127.0.0.1:9",
    NO_PROXY: "127.0.0.1,localhost",
    PYTHONPATH: pythonPath,
    PYTHONUNBUFFERED: "1",
  };
  const comfyOutput = [];
  const productOutput = [];
  const attackerOutput = [];

  try {
    assertNotCancelled();
    attackerProcess = spawnUvicorn(
      python,
      "e2e.fixtures.workflow_editor_attacker_app:app",
      attackerPort,
      {
        ...sharedEnvironment,
        LM_ATELIER_E2E_ATTACKER_READY_TOKEN: readyTokens.attacker,
      },
    );
    const attackerExit = childExit(attackerProcess);
    collectOutput(attackerProcess.stdout, attackerOutput);
    collectOutput(attackerProcess.stderr, attackerOutput);
    await waitForReady(
      `${attackerOrigin}/ready/${readyTokens.attacker}`,
      readyTokens.attacker,
      attackerOutput,
      attackerExit,
    );

    assertNotCancelled();
    comfyProcess = spawnUvicorn(
      python,
      "e2e.fixtures.workflow_editor_comfy_app:app",
      comfyPort,
      {
        ...sharedEnvironment,
        LM_ATELIER_E2E_COMFY_READY_TOKEN: readyTokens.comfy,
      },
    );
    const comfyExit = childExit(comfyProcess);
    collectOutput(comfyProcess.stdout, comfyOutput);
    collectOutput(comfyProcess.stderr, comfyOutput);
    await waitForReady(
      `${comfyOrigin}/ready/${readyTokens.comfy}`,
      readyTokens.comfy,
      comfyOutput,
      comfyExit,
    );

    assertNotCancelled();
    productProcess = spawnUvicorn(
      python,
      "e2e.fixtures.workflow_editor_product_app:app",
      appPort,
      {
        ...sharedEnvironment,
        LOCAL_LM_DATA_DIR: dataDirectory,
        LOCAL_LM_WEB_DIST_DIR: path.join(repositoryRoot, "apps", "web", "dist"),
        LOCAL_LM_HOST: "127.0.0.1",
        LOCAL_LM_PORT: String(appPort),
        LOCAL_LM_DEV: "false",
        LOCAL_LM_CHAT_ENGINE: "mock",
        LOCAL_LM_MEDIA_ENGINE: "comfyui",
        LOCAL_LM_COMFY_URL: comfyOrigin,
        LM_ATELIER_E2E_PRODUCT_READY_TOKEN: readyTokens.product,
      },
    );
    const productExit = childExit(productProcess);
    collectOutput(productProcess.stdout, productOutput);
    collectOutput(productProcess.stderr, productOutput);
    await waitForReady(
      `${baseURL}/api/e2e/workflow-editor-ready/${readyTokens.product}`,
      readyTokens.product,
      productOutput,
      productExit,
    );

    const playwrightCli = require.resolve("@playwright/test/cli");
    assertNotCancelled();
    testProcess = spawn(
      process.execPath,
      [
        playwrightCli,
        "test",
        "e2e/workflow-editor-preview.spec.ts",
        "--config",
        "playwright.config.ts",
      ],
      {
        cwd: repositoryRoot,
        detached: process.platform !== "win32",
        env: {
          ...process.env,
          LM_ATELIER_E2E_BASE_URL: baseURL,
          LM_ATELIER_E2E_COMFY_ORIGIN: comfyOrigin,
          LM_ATELIER_E2E_ATTACKER_ORIGIN: attackerOrigin,
          LM_ATELIER_E2E_OUTPUT_DIR: outputDirectory,
        },
        stdio: "inherit",
        windowsHide: true,
      },
    );
    const result = await childExit(testProcess);
    if (requestedSignal) {
      process.exitCode = requestedSignal === "SIGINT" ? 130 : 143;
    } else if (result.code !== 0) {
      process.exitCode = result.code ?? 1;
      if (attackerOutput.length) {
        process.stderr.write("\nHostile-origin fixture output:\n");
        process.stderr.write(attackerOutput.join(""));
      }
      if (comfyOutput.length) {
        process.stderr.write("\nSynthetic Comfy fixture output:\n");
        process.stderr.write(comfyOutput.join(""));
      }
      if (productOutput.length) {
        process.stderr.write("\nLM Atelier fixture output:\n");
        process.stderr.write(productOutput.join(""));
      }
    }
  } finally {
    if (cancellationCleanup) await cancellationCleanup;
    await terminateOwnedProcesses(Boolean(requestedSignal));
    await validateTemporaryRoot(temporaryRoot);
    await rm(temporaryRoot, { recursive: true, force: true });
  }
}

await main();
