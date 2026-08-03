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
