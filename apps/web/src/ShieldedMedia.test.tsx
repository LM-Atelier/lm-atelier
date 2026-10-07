import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ArtifactPart } from "./ArtifactPart";
import { SENSITIVE_MEDIA_KEY } from "./sensitiveMedia";
import { SensitiveMediaSetting } from "./SensitiveMediaSetting";
import { ShieldedMedia } from "./ShieldedMedia";
import type { MessagePart } from "./types";

function picture(alt = "Generated image") {
  return <ShieldedMedia kind="image"><img src="/api/artifacts/neutral/content" alt={alt} /></ShieldedMedia>;
}

function part(type: MessagePart["type"], name = "neutral-name.png"): MessagePart {
  return {
    id: `part-${type}`,
    position: 0,
    type,
    text: null,
    artifact_id: "sha256:neutral",
    artifact: { original_name: name } as MessagePart["artifact"],
    metadata_json: {},
  };
}

beforeEach(() => localStorage.clear());
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("covering pictures and videos until they are shown", () => {
  it("shows them as it always has until a cover is chosen", () => {
    render(picture());

    expect(screen.getByRole("img", { name: "Generated image" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Show picture" })).toBeNull();
  });

  it("blurs one out of reach of assistive technology and the keyboard until it is shown", () => {
    localStorage.setItem(SENSITIVE_MEDIA_KEY, "blur");
    const { container } = render(picture());

    expect(screen.queryByRole("img")).toBeNull();
    const cover = container.querySelector(".media-shield-cover");
    expect(cover).toHaveAttribute("aria-hidden", "true");
    expect(cover).toHaveAttribute("inert");
    expect(screen.getByText("This picture is blurred.")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Show picture" }));

    expect(screen.getByRole("img", { name: "Generated image" })).toBeVisible();
    expect(container.querySelector(".media-shield")).toBeNull();
  });

  it("does not even load one that is hidden", () => {
    localStorage.setItem(SENSITIVE_MEDIA_KEY, "hide");
    const { container } = render(<ShieldedMedia kind="video"><img src="/api/artifacts/neutral/poster" alt="Generated video" /></ShieldedMedia>);

    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByText("This video is hidden.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Show video" })).toBeVisible();
  });

  it("uncovers only the one shown, and covers it again once it leaves the screen", () => {
    localStorage.setItem(SENSITIVE_MEDIA_KEY, "hide");
    const { unmount } = render(<>{picture("First")}{picture("Second")}</>);

    fireEvent.click(screen.getAllByRole("button", { name: "Show picture" })[0]);

    expect(screen.getByRole("img", { name: "First" })).toBeVisible();
    expect(screen.queryByRole("img", { name: "Second" })).toBeNull();
    unmount();
    render(picture("First"));
    expect(screen.queryByRole("img", { name: "First" })).toBeNull();
  });

  it("follows a choice made in another tab", () => {
    render(picture());

    act(() => {
      localStorage.setItem(SENSITIVE_MEDIA_KEY, "hide");
      window.dispatchEvent(new StorageEvent("storage", { key: SENSITIVE_MEDIA_KEY }));
    });

    expect(screen.queryByRole("img")).toBeNull();
  });

  it("covers a chat's pictures, videos and the names of its attachments", () => {
    localStorage.setItem(SENSITIVE_MEDIA_KEY, "hide");
    const { container } = render(
      <>
        <ArtifactPart part={part("image")} origin="generated" />
        <ArtifactPart part={part("video")} origin="generated" />
        <ArtifactPart part={part("attachment")} origin="uploaded" />
      </>,
    );

    expect(container.querySelector("img, video")).toBeNull();
    expect(screen.getByRole("button", { name: "Show picture" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Show video" })).toBeVisible();
    expect(screen.queryByText("neutral-name.png")).toBeNull();
    expect(screen.getByRole("link", { name: "Attachment" })).toBeVisible();
  });
});

describe("the setting", () => {
  it("is remembered in this browser and applied at once", () => {
    render(<>{picture()}<SensitiveMediaSetting /></>);
    const choices = screen.getByRole("group", { name: "Pictures and videos in chats" });
    expect(screen.getByRole("button", { name: "Show" })).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(screen.getByRole("button", { name: "Blur" }));

    expect(localStorage.getItem(SENSITIVE_MEDIA_KEY)).toBe("blur");
    expect(choices.querySelector('[aria-pressed="true"]')).toHaveTextContent("Blur");
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("says a choice this browser cannot save was not changed", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("storage refused");
    });
    render(<SensitiveMediaSetting />);

    fireEvent.click(screen.getByRole("button", { name: "Hide" }));

    expect(screen.getByRole("status")).toHaveTextContent("This browser cannot save the choice, so it was not changed.");
    expect(screen.getByRole("button", { name: "Show" })).toHaveAttribute("aria-pressed", "true");
  });
});
