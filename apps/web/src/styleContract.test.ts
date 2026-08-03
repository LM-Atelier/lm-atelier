/** Every class the app asks for must exist in the stylesheet.
 *
 * The Image Studio shipped across five pull requests without a single CSS
 * rule: its three canvas layers had no positioning, so the mask painted
 * beside the picture instead of on it, and the selected tool showed nothing.
 * Type checking cannot catch that - a className is just a string - so this
 * checks it directly.
 */

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SOURCE_DIR = join(__dirname);
const STYLESHEET = join(SOURCE_DIR, "styles.css");

/** Classes that are deliberately not styled here, with the reason. */
const UNSTYLED_BY_DESIGN = new Set<string>([]);

function definedClasses(): Set<string> {
  const css = readFileSync(STYLESHEET, "utf8");
  const defined = new Set<string>();
  for (const match of css.matchAll(/\.(-?[_a-zA-Z][\w-]*)/g)) defined.add(match[1]);
  return defined;
}

/** Each element's literal class tokens, paired with the file it came from.
 *
 * An element is checked as a whole rather than token by token. A second
 * token beside a styled base class is a naming hook and styles nothing on
 * its own; an element where *no* token resolves has no styling at all, and
 * that is the defect worth failing on.
 */
function styledElements(): Array<{ tokens: string[]; file: string }> {
  const elements: Array<{ tokens: string[]; file: string }> = [];
  const files = readdirSync(SOURCE_DIR).filter(
    (name) => name.endsWith(".tsx") && !name.endsWith(".test.tsx"),
  );
  for (const name of files) {
    const source = readFileSync(join(SOURCE_DIR, name), "utf8");
    // Both className="a b" and className={`a ${x ? "b" : ""}`}: the literal
    // runs are what matter, since a computed fragment cannot be checked.
    for (const match of source.matchAll(/className=(?:"([^"]*)"|\{`([^`]*)`\})/g)) {
      const literal = (match[1] ?? match[2]).replace(/\$\{[^}]*\}/g, " ");
      const tokens = literal.split(/\s+/).filter((token) => token && !token.endsWith("-"));
      if (tokens.length > 0) elements.push({ tokens, file: name });
    }
  }
  return elements;
}

function relativeLuminance(hex: string): number {
  const value = hex.replace("#", "");
  const channels = [0, 2, 4].map((at) => parseInt(value.slice(at, at + 2), 16) / 255);
  const linear = channels.map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

function contrast(a: string, b: string): number {
  const [high, low] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}

/** A token's value read from the stylesheet, so every check follows the
 *  palette rather than restating it. */
function token(name: string): string {
  const css = readFileSync(STYLESHEET, "utf8");
  return css.match(new RegExp(`--${name}:\\s*(#[0-9a-f]{6})`))?.[1] ?? "";
}

const SURFACES = ["bg", "panel", "panel-2", "panel-3"];

describe("style contract", () => {
  it("leaves no element without any styling at all", () => {
    const defined = definedClasses();
    const unstyled = styledElements()
      .filter(({ tokens }) =>
        tokens.every((token) => !defined.has(token) && !UNSTYLED_BY_DESIGN.has(token)))
      .map(({ tokens, file }) => `${tokens.join(" ")} (${file})`)
      .sort();

    expect([...new Set(unstyled)]).toEqual([]);
  });

  it("stacks the three studio canvas layers in one box", () => {
    const css = readFileSync(STYLESHEET, "utf8");
    // Named explicitly because the failure is silent: inline canvases lay
    // out in a row and the picture still renders, so the mask simply never
    // appears over it.
    expect(css).toMatch(/\.studio-canvas\s*\{[^}]*position:\s*relative/);
    expect(css).toMatch(/\.studio-canvas-layers\s*>\s*canvas\s*\{[^}]*position:\s*absolute/);
  });
});

describe("contrast and state", () => {
  const css = readFileSync(STYLESHEET, "utf8");



  it("keeps muted text readable on every surface it can sit on", () => {
    // Nearly every small label in the app resolves to this one token, and
    // most of it is 9-11px where no large-text exemption applies.
    const worst = Math.min(...SURFACES.map((name) => contrast(token("muted"), token(name))));
    expect(worst).toBeGreaterThanOrEqual(4.5);
  });

  it("gives controls a boundary that can actually be seen", () => {
    // WCAG 1.4.11 wants 3:1 for the edge of anything you type into. The
    // decorative --border measures about 1.4:1 and is a different job.
    const worst = Math.min(...SURFACES.map((name) => contrast(token("line-control"), token(name))));
    expect(worst).toBeGreaterThanOrEqual(3);
  });

  it("does not leave an indeterminate bar parked at a false percentage", () => {
    const reducedMotion = css.slice(css.indexOf("@media (prefers-reduced-motion"));
    expect(reducedMotion).toMatch(/\.indeterminate\s*\{[^}]*width:\s*100%/);
  });

  it("distinguishes where you are from where the pointer is", () => {
    for (const selector of [".primary-nav button.active", ".chat-main.active"]) {
      const rule = css.match(new RegExp(`${selector}[^{]*\\{([^}]*)\\}`))?.[1] ?? "";
      expect(rule).toMatch(/box-shadow/);
    }
  });
});

describe("elevation", () => {

  it("separates each surface from the one below it", () => {
    // Four named levels spanning 1.05 to 1.10 apart are one surface with
    // extra names; nothing in the app read as sitting on anything.
    const ladder = ["bg", "panel", "panel-2", "panel-3"].map(token);
    for (let at = 1; at < ladder.length; at += 1) {
      expect(contrast(ladder[at], ladder[at - 1])).toBeGreaterThan(1.07);
    }
    expect(contrast(ladder[3], ladder[0])).toBeGreaterThan(1.35);
  });
});

describe("the token layer", () => {
  it("does not grow the pile of one-off colours", () => {
    const css = readFileSync(STYLESHEET, "utf8");
    const body = css.slice(css.indexOf("}") + 1);
    const literals = body.match(/#[0-9a-fA-F]{3,8}\b/g) ?? [];

    // A ceiling, not a target. It ratchets down as values earn names; a
    // change needing a new one-off colour is a change needing a token. The
    // palette cannot be reasoned about - or replaced - while most of it is
    // spelled out across the rules.
    expect(literals.length).toBeLessThanOrEqual(155);
  });
});

describe("typography", () => {
  it("never leaves a stack that can fall through to nothing", () => {
    const css = readFileSync(STYLESHEET, "utf8");
    const GENERIC = /^(serif|sans-serif|monospace|cursive|fantasy|system-ui|ui-serif|ui-sans-serif|ui-monospace|inherit)$/;

    // Inter was declared as the interface font and never shipped - no
    // @font-face, no font files anywhere in the repository - so what a
    // reader actually saw depended entirely on what they happened to have
    // installed. A stack is only honest if its last entry is one that
    // every platform can satisfy.
    const endings = [...css.matchAll(/font-family:\s*([^;]+);/g)]
      .map((match) => match[1].split(",").at(-1)!.trim().replace(/^["']|["']$/g, ""))
      .filter((name) => !GENERIC.test(name));

    expect(endings).toEqual([]);
  });
});
