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

describe("the token layer", () => {
  it("does not grow the pile of one-off colours", () => {
    const css = readFileSync(STYLESHEET, "utf8");
    // Custom-property declarations are the palette. A room declaring its
    // own colours is the whole point of having two rooms, so only literals
    // inside ordinary rules count against the ceiling.
    const rules = css.replace(/--[\w-]+:[^;]+;/g, "");
    const literals = rules.match(/#[0-9a-fA-F]{3,8}\b/g) ?? [];

    // A ceiling, not a target. It ratchets down as values earn names; a
    // change needing a new one-off colour is a change needing a token. The
    // palette cannot be reasoned about - or replaced - while most of it is
    // spelled out across the rules.
    expect(literals.length).toBeLessThanOrEqual(65);
  });

  it("paints no surface or ink with a colour that cannot change room", () => {
    // This is the defect the contrast test could not see. Every rule it
    // walked used tokens, so every room looked correct - while `:root`
    // painted `#0f0d0b` under a light ink, the sidebar hardcoded a dark
    // gradient, and a dozen badges spelled out dark-room colours. In the
    // dark room they all agreed and nothing looked wrong. In the light one
    // the chrome stayed dark, the ink stayed pale, and text on paper became
    // unreadable.
    //
    // A literal cannot follow the room. So a surface or an ink must name a
    // token, and only the palette blocks may hold a colour.
    const css = readFileSync(STYLESHEET, "utf8");
    const offenders: string[] = [];
    for (const rule of css.matchAll(/([^{}\n][^{}]*)\{([^}]*)\}/g)) {
      const [, selector, body] = rule;
      if (selector.trim().startsWith("[data-room=")) continue;
      const declarations = body.matchAll(
        /(?<![-\w])(background|background-color|color|border-color):\s*([^;]+);/g,
      );
      for (const [, property, value] of declarations) {
        if (/#[0-9a-fA-F]{3,8}\b|rgba?\(/.test(value)) {
          offenders.push(`${selector.trim().slice(0, 40)} { ${property}: ${value.trim()} }`);
        }
      }
    }

    expect(offenders).toEqual([]);
  });
});

describe("typography", () => {
  it("ships every face it names first", () => {
    const css = readFileSync(STYLESHEET, "utf8");
    const declared = new Set(
      [...css.matchAll(/@font-face\s*\{[^}]*font-family:\s*"?([^";]+)"?;/g)].map((m) =>
        m[1].trim()),
    );
    const files = new Set(readdirSync(join(SOURCE_DIR, "..", "public", "fonts")));

    // A face named first in a stack is the one the design intends. Inter was
    // named that way for a year without a single font file in the repository,
    // so what a reader saw depended on what they happened to have installed.
    expect(declared.size).toBeGreaterThan(0);
    for (const family of declared) {
      const slug = family.toLowerCase().replace(/[^a-z0-9]+/g, "-");
      expect([...files].some((name) => name.startsWith(slug))).toBe(true);
    }
    const leaders = [...css.matchAll(/font-family:\s*([^;]+);/g)]
      .map((m) => m[1].split(",")[0].trim().replace(/^["']|["']$/g, ""))
      .filter((name) => /^[A-Z]/.test(name) && !declared.has(name));
    expect(leaders).toEqual([]);
  });

  it("never leaves a stack that can fall through to nothing", () => {
    // @font-face names one face rather than a stack, so it has nothing to
    // fall through to and nothing to check.
    const css = readFileSync(STYLESHEET, "utf8").replace(/@font-face\s*\{[^}]*\}/g, "");
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

describe("scale and rhythm", () => {
  function stepsOf(property: string): number[] {
    const css = readFileSync(STYLESHEET, "utf8");
    const found = [...css.matchAll(new RegExp(`\\b${property}:\\s*(\\d+)px;`, "g"))];
    return [...new Set(found.map((match) => Number(match[1])))].sort((a, b) => a - b);
  }

  it("keeps type, radii, and spacing on a scale rather than on 17 opinions", () => {
    // Seventeen font sizes, twenty-one radii, and twenty-one gap values are
    // not a system; 7px and 9px gaps sit on no grid at all. Ceilings, so a
    // new arbitrary value has to justify itself as a new step.
    expect(stepsOf("font-size").length).toBeLessThanOrEqual(8);
    expect(stepsOf("border-radius").length).toBeLessThanOrEqual(6);
    expect(stepsOf("gap").length).toBeLessThanOrEqual(7);
  });

  it("leaves the density floor alone", () => {
    // Whether this interface should be denser or airier is a design
    // decision. This change is only about it having a scale at all, so the
    // smallest step must not drift while nobody is looking.
    expect(Math.min(...stepsOf("font-size"))).toBe(9);
  });
});

/** A room's token values, read from the block that declares them. */
function room(selector: string): Record<string, string> {
  const css = readFileSync(STYLESHEET, "utf8");
  const at = css.indexOf(selector);
  const block = css.slice(at, css.indexOf("}", at));
  return Object.fromEntries(
    [...block.matchAll(/--([\w-]+):\s*(#[0-9a-f]{3,8});/g)].map((m) => [m[1], m[2]]),
  );
}

describe("the two rooms", () => {
  const ROOMS = [
    { name: "making", tokens: room(":root {") },
    { name: "reading", tokens: room('[data-room="reading"]') },
  ];

  it.each(ROOMS)("keeps $name readable on every one of its surfaces", ({ tokens }) => {
    const surfaces = ["surface-0", "surface-1", "surface-2", "surface-3"].map((k) => tokens[k]);
    // Against every surface, not just the page. Checking only the lightest
    // is how a muted ink lands at 4.38 on a pressed row and passes review.
    for (const ink of ["ink-primary", "ink-secondary", "ink-muted"]) {
      const worst = Math.min(...surfaces.map((s) => contrast(tokens[ink], s)));
      expect(worst).toBeGreaterThanOrEqual(4.5);
    }
    const line = Math.min(...surfaces.map((s) => contrast(tokens["line-control"], s)));
    expect(line).toBeGreaterThanOrEqual(3);
  });

  it.each(ROOMS)("keeps $name's marks and words each legible enough", ({ tokens }) => {
    const surfaces = ["surface-0", "surface-1", "surface-2", "surface-3"].map((k) => tokens[k]);
    // A mark and a word do not want the same value. 1.4.11 asks 3:1 of a
    // border or an indicator; 1.4.3 asks 4.5:1 of anything you read. One
    // token serving both is how terracotta ended up illegible as text
    // while looking fine as a border.
    for (const mark of ["accent-warm", "info", "state-danger", "state-success"]) {
      expect(Math.min(...surfaces.map((s) => contrast(tokens[mark], s)))).toBeGreaterThanOrEqual(3);
    }
    for (const word of ["accent-warm-text", "state-danger-text", "state-success-text"]) {
      expect(Math.min(...surfaces.map((s) => contrast(tokens[word], s)))).toBeGreaterThanOrEqual(4.5);
    }
    // And the accent as a filled button, which pulls the opposite way: the
    // fill has to be dark enough for the ink that sits on it.
    expect(contrast(tokens["accent-ink"], tokens["accent-fill"])).toBeGreaterThanOrEqual(4.5);
  });

  it.each(ROOMS)("separates $name's surfaces from one another", ({ tokens }) => {
    const ladder = ["surface-0", "surface-1", "surface-2", "surface-3"].map((k) => tokens[k]);
    for (let at = 1; at < ladder.length; at += 1) {
      expect(contrast(ladder[at], ladder[at - 1])).toBeGreaterThan(1.06);
    }
  });

  it("declares the same tokens in both rooms", () => {
    // A token missing from one room silently inherits the other's value,
    // which is how a dark ink ends up on paper.
    const making = room(":root {");
    const reading = room('[data-room="reading"]');
    const themed = Object.keys(reading);
    expect(themed.filter((key) => !(key in making))).toEqual([]);
  });
});

describe("moving between rooms", () => {
  it("puts the work surface in a room and leaves the building alone", () => {
    const app = readFileSync(join(SOURCE_DIR, "App.tsx"), "utf8");
    const rooms = readFileSync(join(SOURCE_DIR, "rooms.ts"), "utf8");
    // The work surface follows the room, and a fixed appearance choice
    // overrides the rooms entirely - someone working at night wants the
    // whole thing dark whatever prose prefers.
    expect(app).toMatch(/data-room=\{roomFor\(theme, READING_ROOM_VIEWS\.has\(view\)\)\}/);
    expect(app).toMatch(/className="app-shell" data-room=/);
    const reading = rooms.match(/READING_ROOM_VIEWS[^=]*=\s*new Set<View>\(\[([^\]]*)\]\)/)?.[1] ?? "";
    expect(reading).toContain('"chat"');
    expect(reading).toContain('"settings"');
    // Colour cannot be judged against a warm ground, so the surfaces where
    // the work is looked at must not become paper.
    expect(reading).not.toContain('"studio"');
    expect(reading).not.toContain('"media"');
  });

  it("gives the ground, the grain, and the header scrim to the room", () => {
    const css = readFileSync(STYLESHEET, "utf8");
    // Any of these left as a literal would half-flip the page: a dark
    // header bar floating over paper, or a light grain over nothing.
    const main = css.match(/(?:^|\n)main \{([^}]*)\}/)?.[1] ?? "";
    expect(main).toMatch(/var\(--surface-0\)/);
    expect(main).toMatch(/var\(--tooth\)/);
    // The grain has to be part of the ground rather than a layer over it. As
    // an absolutely positioned pseudo-element it painted above in-flow
    // content, so the moment it became visible it became visible across
    // every image on the page.
    expect(css).not.toMatch(/main::before/);
    expect(css).toMatch(/\.chat-header \{[^}]*background:\s*var\(--scrim\)/);
    for (const room of [':root {', '[data-room="reading"]']) {
      const at = css.indexOf(room);
      const block = css.slice(at, css.indexOf("}", at));
      expect(block).toMatch(/--scrim:/);
    }
  });
});

describe("action hierarchy", () => {
  const css = readFileSync(STYLESHEET, "utf8");

  /** The body of the rule whose selector list contains this exact selector.
   *  Rules are often grouped - ".new-chat, .primary" - so matching on the
   *  first selector in the list would miss most of them. */
  function rule(selector: string): string {
    for (const match of css.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
      const selectors = match[1].split(",").map((one) => one.trim());
      if (selectors.includes(selector)) return match[2];
    }
    return "";
  }

  it("gives the three levels three different treatments", () => {
    // 105 identical outlined buttons is not a hierarchy. Primary is filled,
    // secondary is outlined, and the quiet level - toolbar and row actions,
    // where four buttons sat in a row with nothing to say which mattered -
    // carries neither until you point at it.
    expect(rule(".primary")).toMatch(/background:\s*var\(--accent-fill\)/);
    expect(rule(".secondary")).toMatch(/border:\s*1px solid var\(--border\)/);
    expect(rule(".secondary.compact-button")).toMatch(/background:\s*transparent/);
    expect(rule(".secondary.compact-button")).toMatch(/border-color:\s*transparent/);
  });

  it("does not paint the primary action with a gradient it cannot vouch for", () => {
    // The old fill ran blue to terracotta, and the terracotta end measured
    // 3.61 against the label sitting on it - so whether a primary button
    // passed depended on where the text fell across the sweep.
    expect(rule(".primary")).not.toMatch(/gradient/);
    for (const room of [":root {", '[data-room="reading"]']) {
      const at = css.indexOf(room);
      const tokens = Object.fromEntries(
        [...css.slice(at, css.indexOf("}", at)).matchAll(/--([\w-]+):\s*(#[0-9a-f]{6});/g)].map(
          (m) => [m[1], m[2]],
        ),
      );
      expect(contrast(tokens["accent-ink"], tokens["accent-fill"])).toBeGreaterThanOrEqual(4.5);
    }
  });
});

describe("every rule, in every room", () => {
  const css = readFileSync(STYLESHEET, "utf8");

  function resolve(value: string, tokens: Record<string, string>): string | null {
    let current: string | undefined = value.trim();
    for (let hop = 0; hop < 6 && current?.startsWith("var("); hop += 1) {
      current = tokens[current.slice(4, current.indexOf(")")).trim().replace(/^--/, "")];
    }
    return current?.startsWith("#") ? current : null;
  }

  const ROOMS = [
    { name: "making", tokens: room(":root {") },
    { name: "reading", tokens: room('[data-room="reading"]') },
  ];

  it.each(ROOMS)("leaves no unreadable text in the $name room", ({ tokens }) => {
    const surfaces = ["surface-0", "surface-1", "surface-2", "surface-3"].map((k) => tokens[k]);
    const unreadable: string[] = [];

    for (const match of css.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
      const selector = match[1].trim();
      const body = match[2];
      if (selector.startsWith("@") || selector.includes("--")) continue;
      const ink = /(?<!-)\bcolor:\s*([^;]+);/.exec(body);
      if (!ink) continue;
      const foreground = resolve(ink[1], tokens);
      if (!foreground) continue;

      // A rule that paints its own ground is judged against that ground;
      // everything else has to survive any surface it might inherit.
      const own = /\bbackground(?:-color)?:\s*([^;]+);/.exec(body);
      const ownFill =
        own && !own[1].includes("gradient")
          ? resolve(own[1].trim().split(/\s+/).at(-1)!, tokens)
          : null;
      const grounds = ownFill ? [ownFill] : surfaces;
      const worst = Math.min(...grounds.map((ground) => contrast(foreground, ground)));
      if (worst < 4.5) unreadable.push(`${selector.slice(0, 60)} (${worst.toFixed(2)})`);
    }

    // Status colours built for a dark room turn invisible on paper. Nine of
    // them shipped that way, including every link in an assistant answer,
    // because nothing checked a rule against the room it would render in.
    expect(unreadable).toEqual([]);
  });
});

describe("material character", () => {
  const css = readFileSync(STYLESHEET, "utf8");

  it("gives the surfaces tooth rather than graph paper", () => {
    // A 28px grid reads as a drafting overlay laid on top. Fibrous noise
    // reads as the surface itself, which is the difference between a page
    // with lines on it and a page made of something.
    const encoded = css.match(/--tooth: url\("data:image\/svg\+xml;base64,([^"]+)"\)/)?.[1] ?? "";
    expect(encoded).not.toBe("");
    // Decoded, because the texture being real is the whole claim: fibrous
    // noise rather than a repeating rule.
    expect(Buffer.from(encoded, "base64").toString()).toMatch(/feTurbulence[^>]*fractalNoise/);
    // The same texture behaving as the material would: darkening the fibre
    // on paper, lifting it on a dark ground.
    expect(css).toMatch(/background-blend-mode: soft-light/);
    expect(css).toMatch(/\[data-room="reading"\] main \{ background-blend-mode: multiply/);
  });

  it("keeps the cut corner for things that hold work", () => {
    // The one shape idea the product has. It marked exactly two elements by
    // accident before, which is not a language.
    expect(css).toMatch(/--cut: /);
    const holders = [".gallery-card", ".engine-card", ".studio-stage", ".recipe-card"];
    for (const holder of holders) {
      const at = css.indexOf(`\n${holder} {`);
      const rule = at < 0 ? "" : css.slice(at, css.indexOf("}", at));
      expect(rule).toMatch(/border-radius: var\(--cut\)/);
    }
    // A control is not a thing that holds work, so it keeps even corners.
    const secondary = css.match(/\n\.secondary \{([^}]*)\}/)?.[1] ?? "";
    expect(secondary).not.toMatch(/--cut/);
  });

  it("moves quickly and never overshoots", () => {
    // An overshoot on every hover is what makes an interface feel like a
    // toy rather than a bench.
    expect(css).toMatch(/--ease: cubic-bezier\(\.2, \.8, \.2, 1\)/);
    const easing = css.match(/--ease: ([^;]+);/)?.[1] ?? "";
    const [, , third] = easing.match(/cubic-bezier\(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\)/) ?? [];
    expect(Number(third)).toBeLessThanOrEqual(1);
  });
});
