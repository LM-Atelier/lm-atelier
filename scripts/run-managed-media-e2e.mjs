// Certify the surfaces that only exist once a media engine is really MANAGED.
//
// A workflow revision becomes trusted by being reviewed against node information
// fetched from a media worker the product started, is running, has marked ready,
// and can name a process for. An engine the runner starts beside the product
// satisfies none of that, so in the ordinary browser run nothing can be trusted
// and every surface gated on trust - the generation settings panel among them -
// renders nothing to assert against.
//
// So this run does not start an engine. It stages one where a real installation
// would be and lets the product's own media launch start it, which leaves the
// trust path completely unmodified: what is synthetic is the program at the end
// of the launch, not the launch, the review, or the decision.

import { spawn } from "node:child_process";
import { copyFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import process from "node:process";

import {
  OwnedProcesses,
  childExit,
  collectOutput,
  makeTemporaryRoot,
  pythonExecutable,
  readinessToken,
  removeTemporaryRoot,
  repositoryRoot,
  reserveLoopbackPort,
  sharedEnvironment,
  spawnUvicorn,
  waitForReady,
} from "./isolated-e2e-harness.mjs";

const require = createRequire(import.meta.url);
const owned = new OwnedProcesses();

const SPECS = ["e2e/generation-settings-shape.spec.ts"];

/**
 * Put the synthetic engine where the product expects an installation.
 *
 * The launch resolves the directory and requires main.py inside it, so the
 * fixture is COPIED under that name rather than imported: the process the
 * product starts inherits none of the repository's import path, by design, and
 * a stand-in that needed it would be relying on something the real engine does
 * not have either.
 */
async function stageEngine(temporaryRoot) {
  const directory = path.join(temporaryRoot, "engine");
  await mkdir(directory, { recursive: true });
  await copyFile(
    path.join(repositoryRoot, "e2e", "fixtures", "managed_media_engine.py"),
    path.join(directory, "main.py"),
  );
  return directory;
}

async function main() {
  owned.installSignalHandlers();
  const temporaryRoot = await makeTemporaryRoot();
  const dataDirectory = path.join(temporaryRoot, "data");
  // Inside the run's own directory by default, so a passing run leaves nothing
  // behind; overridable because a failing run's traces and accessibility
  // snapshots are the only way to see what the browser actually had, and they
  // are deleted with everything else.
  const outputDirectory = process.env.LM_ATELIER_E2E_OUTPUT_DIR
    ?? path.join(temporaryRoot, "playwright-output");
  const engineDirectory = await stageEngine(temporaryRoot);

  const appPort = await reserveLoopbackPort();
  const enginePort = await reserveLoopbackPort();
  const baseURL = `http://127.0.0.1:${appPort}`;
  const engineOrigin = `http://127.0.0.1:${enginePort}`;
  const productReadyToken = readinessToken();
  const python = await pythonExecutable();
  const productOutput = [];

  try {
    owned.assertNotCancelled("Managed media certification");
    // Owned as a TREE: this product starts the media engine itself, and that
    // engine is nobody else's to stop. Left behind it keeps the run's temporary
    // directory open and the cleanup fails.
    const productProcess = owned.own(spawnUvicorn(
      python,
      "e2e.fixtures.managed_media_product_app:app",
      appPort,
      sharedEnvironment({
        LOCAL_LM_DATA_DIR: dataDirectory,
        LOCAL_LM_WEB_DIST_DIR: path.join(repositoryRoot, "apps", "web", "dist"),
        LOCAL_LM_HOST: "127.0.0.1",
        LOCAL_LM_PORT: String(appPort),
        LOCAL_LM_DEV: "false",
        LOCAL_LM_CHAT_ENGINE: "mock",
        LOCAL_LM_MEDIA_ENGINE: "comfyui",
        // The three that make the engine the product's own to start.
        LOCAL_LM_COMFY_URL: engineOrigin,
        LOCAL_LM_COMFY_EXECUTABLE: python,
        LOCAL_LM_COMFY_DIRECTORY: engineDirectory,
        LM_ATELIER_E2E_PRODUCT_READY_TOKEN: productReadyToken,
      }),
    ), { tree: true });
    const productExit = childExit(productProcess);
    collectOutput(productProcess.stdout, productOutput);
    collectOutput(productProcess.stderr, productOutput);
    await waitForReady(
      `${baseURL}/api/e2e/ready/${productReadyToken}`,
      productReadyToken,
      productOutput,
      productExit,
      "LM Atelier fixture",
    );

    const playwrightCli = require.resolve("@playwright/test/cli");
    owned.assertNotCancelled("Managed media certification");
    const testProcess = owned.own(spawn(
      process.execPath,
      [playwrightCli, "test", ...SPECS, "--config", "playwright.config.ts"],
      {
        cwd: repositoryRoot,
        detached: process.platform !== "win32",
        env: {
          ...process.env,
          LM_ATELIER_E2E_BASE_URL: baseURL,
          LM_ATELIER_E2E_MANAGED_MEDIA: "1",
          LM_ATELIER_E2E_OUTPUT_DIR: outputDirectory,
        },
        stdio: "inherit",
        windowsHide: true,
      },
    ));
    const result = await childExit(testProcess);
    if (owned.requestedSignal) {
      process.exitCode = owned.requestedSignal === "SIGINT" ? 130 : 143;
    } else if (result.code !== 0) {
      process.exitCode = result.code ?? 1;
      if (productOutput.length) {
        process.stderr.write("\nLM Atelier fixture output:\n");
        process.stderr.write(productOutput.join(""));
      }
    }
  } finally {
    if (owned.cancellationCleanup) await owned.cancellationCleanup;
    await owned.terminateAll(Boolean(owned.requestedSignal));
    await removeTemporaryRoot(temporaryRoot);
  }
}

await main();
