/** The composer has to fit the prompt, not the other way round. */

import { useRef, useState } from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MessageField } from "./MessageField";

function Harness({ onSubmit = vi.fn() }: { onSubmit?: () => void }) {
  const field = useRef<HTMLTextAreaElement | null>(null);
  const [text, setText] = useState("");
  return <MessageField field={field} value={text} onChange={setText} onSubmit={onSubmit} />;
}

/** jsdom lays nothing out, so scrollHeight is whatever we say it is. */
function measureAs(field: HTMLTextAreaElement, height: () => number) {
  Object.defineProperty(field, "scrollHeight", { configurable: true, get: height });
}

describe("MessageField", () => {
  afterEach(() => {
    cleanup();
  });

  it("grows to fit what was typed and shrinks when it is deleted", () => {
    render(<Harness />);
    const field = screen.getByLabelText("Message") as HTMLTextAreaElement;
    // Only reports the smaller size when the field was collapsed first, so
    // this fails if the reset before measuring is ever dropped.
    measureAs(field, () => (field.style.height === "auto" && field.value.length < 20 ? 42 : 96));

    fireEvent.change(field, { target: { value: "a prompt long enough to wrap onto more lines" } });
    expect(field.style.height).toBe("96px");

    fireEvent.change(field, { target: { value: "short" } });
    expect(field.style.height).toBe("42px");
  });

  it("sends on Enter and takes a new line on Shift+Enter", () => {
    const onSubmit = vi.fn();
    render(<Harness onSubmit={onSubmit} />);
    const field = screen.getByLabelText("Message");

    fireEvent.keyDown(field, { key: "Enter", shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();

    fireEvent.keyDown(field, { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("still hands the element to the caller's ref", () => {
    function Capturing() {
      const field = useRef<HTMLTextAreaElement | null>(null);
      return (
        <>
          <MessageField field={field} value="" onChange={vi.fn()} onSubmit={vi.fn()} />
          <button onClick={() => field.current?.focus()}>Focus</button>
        </>
      );
    }
    render(<Capturing />);
    // The composer focuses this field from elsewhere; measuring it locally
    // must not cost the caller its handle on the element.
    fireEvent.click(screen.getByRole("button", { name: "Focus" }));
    expect(document.activeElement).toBe(screen.getByLabelText("Message"));
  });
});
