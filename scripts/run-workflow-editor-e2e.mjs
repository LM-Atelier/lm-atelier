// Certify the native workflow editor's browser protocol against a hostile origin.
//
// Three services on three loopback ports: the product, a synthetic ComfyUI
// serving the editor page and the staged bridge, and an unrelated origin that
// tries to talk to it. The engine here is deliberately NOT managed by the
// product - this run is about what a page in a browser may do, and the product
// only needs to know where the editor lives.
//
// The run that needs a MANAGED engine is run-managed-media-e2e.mjs; the shared
// machinery both use is in isolated-e2e-harness.mjs.

import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
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

async function main() {
  owned.installSignalHandlers();
  const temporaryRoot = await makeTemporaryRoot();
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
  const environment = sharedEnvironment({
    LM_ATELIER_E2E_APP_ORIGIN: baseURL,
    LM_ATELIER_E2E_ATTACKER_ORIGIN: attackerOrigin,
    LM_ATELIER_E2E_FIXTURE_ROOT: fixtureDirectory,
  });
  const comfyOutput = [];
  const productOutput = [];
  const attackerOutput = [];

  try {
    owned.assertNotCancelled("Browser protocol certification");
    const attackerProcess = owned.own(spawnUvicorn(
      python,
      "e2e.fixtures.workflow_editor_attacker_app:app",
      attackerPort,
      { ...environment, LM_ATELIER_E2E_ATTACKER_READY_TOKEN: readyTokens.attacker },
    ));
    const attackerExit = childExit(attackerProcess);
    collectOutput(attackerProcess.stdout, attackerOutput);
    collectOutput(attackerProcess.stderr, attackerOutput);
    await waitForReady(
      `${attackerOrigin}/ready/${readyTokens.attacker}`,
      readyTokens.attacker,
      attackerOutput,
      attackerExit,
      "Hostile-origin fixture",
    );

    owned.assertNotCancelled("Browser protocol certification");
    const comfyProcess = owned.own(spawnUvicorn(
      python,
      "e2e.fixtures.workflow_editor_comfy_app:app",
      comfyPort,
      { ...environment, LM_ATELIER_E2E_COMFY_READY_TOKEN: readyTokens.comfy },
    ));
    const comfyExit = childExit(comfyProcess);
    collectOutput(comfyProcess.stdout, comfyOutput);
    collectOutput(comfyProcess.stderr, comfyOutput);
    await waitForReady(
      `${comfyOrigin}/ready/${readyTokens.comfy}`,
      readyTokens.comfy,
      comfyOutput,
      comfyExit,
      "Synthetic Comfy fixture",
    );

    owned.assertNotCancelled("Browser protocol certification");
    const productProcess = owned.own(spawnUvicorn(
      python,
      "e2e.fixtures.workflow_editor_product_app:app",
      appPort,
      {
        ...environment,
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
    ));
    const productExit = childExit(productProcess);
    collectOutput(productProcess.stdout, productOutput);
    collectOutput(productProcess.stderr, productOutput);
    await waitForReady(
      `${baseURL}/api/e2e/ready/${readyTokens.product}`,
      readyTokens.product,
      productOutput,
      productExit,
      "LM Atelier fixture",
    );

    const playwrightCli = require.resolve("@playwright/test/cli");
    owned.assertNotCancelled("Browser protocol certification");
    const testProcess = owned.own(spawn(
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
    ));
    const result = await childExit(testProcess);
    if (owned.requestedSignal) {
      process.exitCode = owned.requestedSignal === "SIGINT" ? 130 : 143;
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
    if (owned.cancellationCleanup) await owned.cancellationCleanup;
    await owned.terminateAll(Boolean(owned.requestedSignal));
    await removeTemporaryRoot(temporaryRoot);
  }
}

await main();
