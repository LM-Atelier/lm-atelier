import { spawn } from "node:child_process";
import { access, mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const TEMP_PREFIX = "lm-atelier-e2e-";
const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let appProcess;
let testProcess;
let requestedSignal;

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
  const environmentPython = process.platform === "win32"
    ? path.join(repositoryRoot, ".venv", "Scripts", "python.exe")
    : path.join(repositoryRoot, ".venv", "bin", "python");
  const projectPython = process.platform === "win32"
    ? path.join(repositoryRoot, "services", "api", ".venv", "Scripts", "python.exe")
    : path.join(repositoryRoot, "services", "api", ".venv", "bin", "python");
  const discovered = await firstExistingPath([environmentPython, projectPython]);
  return discovered ?? (process.platform === "win32" ? "python.exe" : "python3");
}

async function waitForReady(baseURL, appOutput, appExit) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const exited = await Promise.race([
      appExit.then((result) => ({ exited: true, result })),
      new Promise((resolve) => setTimeout(() => resolve({ exited: false }), 150)),
    ]);
    if (exited.exited) {
      throw new Error(
        `LM Atelier exited before becoming ready (${JSON.stringify(exited.result)}).\n`
        + appOutput.join(""),
      );
    }
    try {
      const response = await fetch(`${baseURL}/api/ready`);
      if (response.ok) return;
    } catch {
      // The loopback listener is still starting.
    }
  }
  throw new Error(`LM Atelier did not become ready within 30 seconds.\n${appOutput.join("")}`);
}

function collectOutput(stream, output) {
  stream?.setEncoding("utf8");
  stream?.on("data", (chunk) => {
    output.push(chunk);
    if (output.length > 200) output.shift();
  });
}

async function terminate(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  if (!child.pid) return;
  const exit = childExit(child);
  try {
    child.kill("SIGTERM");
  } catch {
    return;
  }
  const stopped = await Promise.race([
    exit.then(() => true),
    new Promise((resolve) => setTimeout(() => resolve(false), 5_000)),
  ]);
  if (stopped) return;
  if (process.platform === "win32") {
    const killer = spawn(
      "taskkill.exe",
      ["/PID", String(child.pid), "/T", "/F"],
      { windowsHide: true, stdio: "ignore" },
    );
    await childExit(killer);
  } else {
    child.kill("SIGKILL");
  }
  await exit;
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
      testProcess?.kill(signal);
      appProcess?.kill(signal);
    });
  }
}

async function main() {
  installSignalHandlers();
  const temporaryRoot = await mkdtemp(path.join(tmpdir(), TEMP_PREFIX));
  await validateTemporaryRoot(temporaryRoot);
  const dataDirectory = path.join(temporaryRoot, "data");
  const fixtureDirectory = path.join(temporaryRoot, "fixtures");
  const outputDirectory = path.join(temporaryRoot, "playwright-output");
  await mkdir(fixtureDirectory, { recursive: true });

  const modelPath = path.join(fixtureDirectory, "tiny-safe-fixture.gguf");
  const ggufHeader = Buffer.alloc(24);
  ggufHeader.write("GGUF", 0, "ascii");
  ggufHeader.writeUInt32LE(3, 4);
  await writeFile(modelPath, ggufHeader);

  const port = await reserveLoopbackPort();
  const baseURL = `http://127.0.0.1:${port}`;
  const python = await pythonExecutable();
  const appOutput = [];
  const appEnvironment = {
    ...process.env,
    LOCAL_LM_DATA_DIR: dataDirectory,
    LOCAL_LM_WEB_DIST_DIR: path.join(repositoryRoot, "apps", "web", "dist"),
    LOCAL_LM_HOST: "127.0.0.1",
    LOCAL_LM_PORT: String(port),
    LOCAL_LM_DEV: "false",
    LOCAL_LM_CHAT_ENGINE: "mock",
    LOCAL_LM_MEDIA_ENGINE: "mock",
    LOCAL_LM_HF_TOKEN: "e2e-placeholder-not-a-real-token",
    HTTP_PROXY: "http://127.0.0.1:9",
    HTTPS_PROXY: "http://127.0.0.1:9",
    NO_PROXY: "127.0.0.1,localhost",
    PYTHONUNBUFFERED: "1",
  };

  try {
    appProcess = spawn(
      python,
      [
        "-m",
        "uvicorn",
        "local_lm.main:app",
        "--app-dir",
        path.join(repositoryRoot, "services", "api"),
        "--host",
        "127.0.0.1",
        "--port",
        String(port),
      ],
      {
        cwd: repositoryRoot,
        env: appEnvironment,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    const appExit = childExit(appProcess);
    collectOutput(appProcess.stdout, appOutput);
    collectOutput(appProcess.stderr, appOutput);
    await waitForReady(baseURL, appOutput, appExit);

    const playwrightCli = path.join(
      repositoryRoot,
      "node_modules",
      "@playwright",
      "test",
      "cli.js",
    );
    testProcess = spawn(
      process.execPath,
      [playwrightCli, "test", "--config", "playwright.config.ts", ...process.argv.slice(2)],
      {
        cwd: repositoryRoot,
        env: {
          ...process.env,
          LM_ATELIER_E2E_BASE_URL: baseURL,
          LM_ATELIER_E2E_MODEL_PATH: modelPath,
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
      if (appOutput.length) {
        process.stderr.write("\nLM Atelier server output:\n");
        process.stderr.write(appOutput.join(""));
      }
    }
  } finally {
    await terminate(testProcess);
    await terminate(appProcess);
    await validateTemporaryRoot(temporaryRoot);
    await rm(temporaryRoot, { recursive: true, force: true });
  }
}

await main();
