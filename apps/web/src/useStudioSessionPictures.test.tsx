/** The second picture an edit is given: uploaded when it is new, named when the library holds it. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", async (original) => ({
  ...(await original<typeof import("./api")>()),
  api: { openStudioSession: vi.fn(), studioSession: vi.fn(), sendTurn: vi.fn(), upload: vi.fn() },
}));

const SESSION = { id: "chat-studio", messages: [] };
const WORDS = "Replace the subject with the one in the second picture. Keep everything around it exactly as it is.";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.clearAllMocks();
});

async function openStudio() {
  vi.mocked(api.openStudioSession).mockResolvedValue(SESSION as never);
  vi.mocked(api.studioSession).mockResolvedValue(SESSION as never);
  vi.mocked(api.sendTurn).mockResolvedValue({ assistant_message: { id: "msg-1" } } as never);
  const hook = renderHook(() => useStudioSession("art-1", null), { wrapper });
  await waitFor(() => expect(hook.result.current.sessionId).toBe("chat-studio"));
  return hook;
}

it("sends a picture the library already holds as that picture, uploading nothing", async () => {
  const { result } = await openStudio();

  act(() => result.current.apply(WORDS, "art-1", undefined, undefined, "wfrev_two", undefined, "art-library"));

  await waitFor(() => expect(api.sendTurn).toHaveBeenCalledTimes(1));
  expect(api.upload).not.toHaveBeenCalled();
  expect(api.sendTurn).toHaveBeenCalledWith(
    "chat-studio", WORDS, "image", ["art-1", "art-library"], {}, undefined, undefined, "wfrev_two",
  );
});

it("uploads a chosen file and sends the stored copy after the source", async () => {
  const { result } = await openStudio();
  vi.mocked(api.upload).mockResolvedValue({ id: "sha256:chosen" } as never);
  const picture = new File(["neutral"], "new-subject.png", { type: "image/png" });

  act(() => result.current.apply(WORDS, "art-1", undefined, undefined, "wfrev_two", undefined, picture));

  await waitFor(() => expect(api.sendTurn).toHaveBeenCalledTimes(1));
  expect(api.upload).toHaveBeenCalledWith(picture);
  expect(vi.mocked(api.sendTurn).mock.calls[0][3]).toEqual(["art-1", "sha256:chosen"]);
});
