import {
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import { MentionPicker } from "./MentionPicker";
import { insertMention, mentionQuery, type TrackedMention } from "./mentionDraft";
import type { ReferenceSubject } from "./types";

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
  onMention,
}: {
  field: RefObject<HTMLTextAreaElement | null>;
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  /** Called when a subject is chosen from the picker. Omit to disable
   *  mentions entirely - every other caller of this field is unaffected. */
  onMention?: (mention: TrackedMention) => void;
}) {
  // Null while no mention is being written, which is also what closes the
  // picker. Empty string is a real value: "@" alone means show everything.
  const [query, setQuery] = useState<string | null>(null);
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

  const choose = (subject: ReferenceSubject) => {
    const element = own.current;
    const caret = element?.selectionStart ?? value.length;
    const written = insertMention(value, caret, subject.mention_slug);
    onChange(written.text);
    onMention?.({
      referenceSubjectId: subject.id,
      mentionSlug: subject.mention_slug,
    });
    setQuery(null);
    // Put the caret after what was inserted, once React has written the value.
    requestAnimationFrame(() => {
      if (!element) return;
      element.focus();
      element.setSelectionRange(written.caret, written.caret);
    });
  };

  const track = (element: HTMLTextAreaElement) => {
    setQuery(onMention ? mentionQuery(element.value, element.selectionStart ?? 0) : null);
  };

  return (
    <div className="message-field">
      <textarea
        ref={own}
        rows={1}
        aria-label="Message"
        value={value}
        onChange={(event) => {
          onChange(event.target.value);
          track(event.currentTarget);
        }}
        // The caret moves without the text changing, so the picker has to
        // follow selection as well as input or it stays open over a mention
        // the person has already walked away from.
        onSelect={(event) => track(event.currentTarget)}
        onBlur={() => setQuery(null)}
        onKeyDown={(event) => {
          if (event.key === "Escape" && query !== null) {
            event.preventDefault();
            setQuery(null);
            return;
          }
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            onSubmit();
          }
        }}
        placeholder="Ask anything, or describe an image or video to create…"
      />
      {/* Only when the caller opted in. The picker queries the reference
          library, and a field that was never asked to support mentions should
          not acquire that dependency - nor should its callers. */}
      {onMention && <MentionPicker query={query} onChoose={choose} />}
    </div>
  );
}
