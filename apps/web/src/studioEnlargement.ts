/** What an enlargement will do, as the workflow it runs on says, before Apply is pressed.
 *
 * The Studio asks the server which workflow an Enhance of this picture would
 * run and what that workflow lets the person choose. A workflow whose graph
 * applies a chosen factor answers with that factor's field; one that always
 * enlarges by the same amount answers with that amount when its graph proves
 * it, or with nothing when only the workflow knows. The Studio never offers a
 * size the workflow would not make.
 */

import type { EnlargementPreview, SettingField } from "./types";

export type { EnlargementPreview };

/** A factor the person chose, and the preview it was chosen under. */
export type StudioUpscaleChoice = { preview: string; factor: number };

/** The first and last factors a number control offers, and the step between them. */
export type EnlargementScale = { minimum: number | null; maximum: number | null; step: number | "any" };

/** Which preview a choice belongs to: the workflow it named and the field it offered.
 *
 * A choice made under one answer says nothing about another: a new workflow,
 * or a recipe that changes the field's default, starts from the new default.
 */
export function enlargementIdentity(preview: EnlargementPreview): string {
  return `${preview.workflow_revision_id}\u0000${JSON.stringify(preview.factor)}`;
}

/** The factor the person chose under this preview, or null when they chose none under it. */
export function chosenFactor(preview: EnlargementPreview, choice: StudioUpscaleChoice | null): number | null {
  return choice && choice.preview === enlargementIdentity(preview) ? choice.factor : null;
}

/** The factors a field offers as separate choices, or null when it is a number. */
export function enlargementChoices(field: SettingField | null): number[] | null {
  if (!field || field.choices.length === 0) return null;
  const numbers = field.choices.filter((choice): choice is number => typeof choice === "number" && Number.isFinite(choice));
  return numbers.length > 0 ? [...new Set(numbers)].sort((left, right) => left - right) : null;
}

/** The factor to send: the one chosen when the field takes it, otherwise the field's own default.
 *
 * Null when the field offers no factor, or offers a number with neither a
 * default nor a choice, so the workflow's own value stands. A choice from a
 * list must be one of the list. A number keeps to the bounds the field
 * declares, and to its multiple counted from zero, as the server checks it,
 * so a turn is not refused after Apply for a factor the panel allowed.
 */
export function enlargementFactor(field: SettingField | null, chosen: number | null): number | null {
  if (!field || !field.available) return null;
  const fallback = typeof field.default === "number" && Number.isFinite(field.default) ? field.default : null;
  const choices = enlargementChoices(field);
  if (choices) {
    if (chosen !== null && choices.includes(chosen)) return chosen;
    return fallback !== null && choices.includes(fallback) ? fallback : choices[0];
  }
  const wanted = chosen ?? fallback;
  return wanted === null ? null : legalNumber(field, wanted);
}

/** How a number control moves: its first and last legal factors, and the step between them.
 *
 * With a multiple, the step is the one a factor is kept to and the ends are
 * legal factors themselves, so a slider or number box steps only through
 * factors the workflow takes, counted from zero as the server counts them
 * rather than from a declared minimum that is no multiple. Without one, the
 * field's own step is kept as the control's, where it suits the field. A bound
 * the field does not declare stays open. Null when no factor within the bounds
 * is legal.
 */
export function enlargementScale(field: SettingField): EnlargementScale | null {
  const unit = factorUnit(field);
  if (unit === null) return null;
  const minimum = field.minimum === null ? null : legalNumber(field, field.minimum);
  const maximum = field.maximum === null ? null : legalNumber(field, field.maximum);
  if ((field.minimum !== null && minimum === null) || (field.maximum !== null && maximum === null)) return null;
  const own = field.step;
  const suits = own !== null && Number.isFinite(own) && own > 0 && (field.type !== "integer" || Number.isInteger(own));
  return { minimum, maximum, step: positiveMultiple(field) === null && suits ? own : unit };
}

/** A number moved onto what the field declares: its bounds, and the steps of its unit.
 *
 * Null when no step lies within the bounds, and never a value the server's
 * own check would refuse.
 */
function legalNumber(field: SettingField, value: number): number | null {
  const unit = factorUnit(field);
  if (unit === null) return null;
  const { minimum, maximum } = field;
  let result = Math.min(maximum ?? Infinity, Math.max(minimum ?? -Infinity, value));
  if (unit !== "any") {
    result = Math.round(result / unit) * unit;
    if (minimum !== null && result < minimum) result = Math.ceil(minimum / unit) * unit;
    if (maximum !== null && result > maximum) result = Math.floor(maximum / unit) * unit;
  }
  // A multiple such as 0.2 leaves binary noise behind: 0.6000000000000001 is 0.6.
  result = Number(result.toFixed(10));
  const inside = (minimum === null || result >= minimum) && (maximum === null || result <= maximum);
  const multiple = positiveMultiple(field);
  const whole = field.type !== "integer" || Number.isInteger(result);
  return inside && whole && (multiple === null || isMultiple(result, multiple)) ? result : null;
}

/** The steps a factor moves in: its multiple, whole numbers, or both at once; "any" for neither.
 *
 * A whole-number field with a multiple moves in the least whole number that is
 * also one of its multiples: whole multiples of 1.5 are the multiples of
 * three, and whole multiples of 0.3 too. Null when there is none to find.
 */
function factorUnit(field: SettingField): number | "any" | null {
  const multiple = positiveMultiple(field);
  if (field.type !== "integer") return multiple ?? "any";
  return multiple === null ? 1 : leastWholeMultiple(multiple);
}

/** The multiple the field declares, when it declares one the server would check against. */
function positiveMultiple(field: SettingField): number | null {
  return field.multiple_of && field.multiple_of > 0 ? field.multiple_of : null;
}

/** Whether the server counts a value a multiple: its quotient within a billionth of a whole number. */
function isMultiple(value: number, multiple: number): boolean {
  const quotient = value / multiple;
  const nearest = Math.round(quotient);
  return Math.abs(quotient - nearest) <= Math.max(1e-9 * Math.max(Math.abs(quotient), Math.abs(nearest)), 1e-9);
}

/** The least whole number that is a multiple, or null when none is found.
 *
 * The continued fraction of the multiple gives its best approximations in
 * order of size, so the first numerator that passes as a multiple is the
 * least whole one: 3 for 1.5 (3/2) and for 0.3 (3/10), 1 for 0.25 (1/4).
 */
function leastWholeMultiple(multiple: number): number | null {
  let [numerator, previous] = [1, 0];
  let rest = multiple;
  for (let term = 0; term < 64 && Number.isFinite(rest); term += 1) {
    const whole = Math.floor(rest);
    [numerator, previous] = [whole * numerator + previous, numerator];
    if (numerator > 0 && isMultiple(numerator, multiple)) return numerator;
    rest = 1 / (rest - whole);
  }
  return null;
}

/** What the Enhance panel says about the size, when the person has nothing to choose. */
export function describeFixedEnlargement(preview: EnlargementPreview): string {
  return preview.fixed_factor !== null
    ? `Enlarges ${preview.fixed_factor}x, set by the workflow.`
    : "The workflow sets how much it enlarges.";
}
