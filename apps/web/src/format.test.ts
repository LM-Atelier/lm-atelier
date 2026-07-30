import { describe, expect, it } from "vitest";
import { docsLink, supportLinks } from "./format";

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
});
