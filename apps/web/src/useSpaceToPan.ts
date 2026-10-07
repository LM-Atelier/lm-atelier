import { useEffect, useRef } from "react";

/** Whether Space is held to pan, read from the whole window.
 *
 * A Space or a mouse button released in another window is never heard here,
 * so losing focus counts as letting go of both: Space is read as up, and
 * `onFocusLost` ends whatever the canvas was in the middle of.
 */
export function useSpaceToPan(onFocusLost: () => void) {
  const spaceHeld = useRef(false);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Space") spaceHeld.current = event.type === "keydown";
    };
    const onBlur = () => {
      spaceHeld.current = false;
      onFocusLost();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
      window.removeEventListener("blur", onBlur);
    };
  }, [onFocusLost]);
  return spaceHeld;
}
