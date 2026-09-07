import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { docsLink, supportLinks } from "./format";

const REPOSITORY_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../..",
);

/**
 * Every path git has, exactly as git spells it.
 *
 * `existsSync` is the wrong instrument here: this machine's filesystem is
 * case-insensitive, so it answers yes for `docs/getting-started.md` when the
 * file is really `docs/GETTING-STARTED.md`. The link is a GitHub blob URL and
 * GitHub is case-sensitive, so that answer would be wrong in the one way that
 * matters. Records are NUL-separated because a path may contain a space.
 */
function trackedPaths(): Set<string> {
  const records = execFileSync("git", ["ls-files", "-z"], {
    cwd: REPOSITORY_ROOT,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  return new Set(records.split("\0").filter((path) => path.length > 0));
}

describe("docsLink", () => {
  it("pins documentation to the running release", () => {
    expect(docsLink("0.1.8", "docs/TROUBLESHOOTING.md")).toBe(
      "https://github.com/ajccarlson/lm-atelier/blob/v0.1.8/docs/TROUBLESHOOTING.md",
    );
  });

  it("falls back to the branch when the version is not a release", () => {
    // A broken link helps nobody, so an unrecognised version is not guessed at.
    expect(docsLink("0.1.8-dev", "SUPPORT.md")).toBe(
      "https://github.com/ajccarlson/lm-atelier/blob/main/SUPPORT.md",
    );
  });
});

describe("supportLinks", () => {
  it("offers troubleshooting first, because that is why people look", () => {
    expect(supportLinks("0.1.8")[0][0]).toBe("Troubleshooting");
  });

  it("includes the issue tracker unpinned", () => {
    const issues = supportLinks("0.1.8").find(([label]) => label === "Issues");
    expect(issues?.[1]).toBe("https://github.com/ajccarlson/lm-atelier/issues");
  });

  it("only offers documents this repository actually has", () => {
    // Every help destination but the issue tracker is a document in this
    // repository, linked by exact path. Rename or remove one and the menu
    // still renders: the reader clicks Privacy and gets a 404, which is the
    // worst moment to find out, because they went looking for help. Nothing
    // was holding those five paths to anything, so this does.
    const tracked = trackedPaths();
    const documents = supportLinks("0.1.8")
      .map(([, url]) => url)
      .filter((url) => url.includes("/blob/"))
      .map((url) => url.split("/blob/v0.1.8/")[1]);

    expect(documents).not.toHaveLength(0);
    expect(documents.filter((path) => !tracked.has(path))).toEqual([]);
  });
});
