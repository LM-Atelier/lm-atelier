import { useImperativeHandle, useLayoutEffect, useRef, type RefObject } from "react";

/** The composer's text field, sized to whatever is being written in it.
 *
 * A single-row box is the wrong shape for this product: describing an image
 * is the central task and those descriptions run to paragraphs. The height
 * is measured rather than counted in rows, because wrapping - not line
 * breaks - is usually what makes a prompt long. The ceiling lives in CSS,
 * and past it the field scrolls, which is the point where growing further
 * would start eating the conversation.
 */
export function MessageField({
  field,
  value,
  onChange,
  onSubmit,
}: {
  field: RefObject<HTMLTextAreaElement | null>;
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}) {
  const own = useRef<HTMLTextAreaElement | null>(null);
  // The composer focuses this field from several places, so the caller keeps
  // its handle - handed over through the sanctioned path rather than by
  // writing to a ref this component was given.
  useImperativeHandle<HTMLTextAreaElement | null, HTMLTextAreaElement | null>(
    field,
    () => own.current,
    [],
  );

  useLayoutEffect(() => {
    const element = own.current;
    if (!element) return;
    // Collapse before measuring: scrollHeight cannot report less than the
    // height already set, so without this the field would keep its
    // high-water mark and never shrink back down.
    element.style.height = "auto";
    element.style.height = `${element.scrollHeight}px`;
  }, [value]);

  return (
    <textarea
      ref={own}
      rows={1}
      aria-label="Message"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          onSubmit();
        }
      }}
      placeholder="Ask anything, or describe an image or video to create…"
    />
  );
}
