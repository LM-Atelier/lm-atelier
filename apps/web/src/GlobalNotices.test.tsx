import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { GlobalNotices } from "./GlobalNotices";

const ok = { error: null };
const failed = (message: string) => ({ error: new Error(message) });

describe("GlobalNotices", () => {
  afterEach(cleanup);

  it("reports a failure from anywhere in the list", () => {
    // The hand-written `||` chains this replaces were duplicated and had to be
    // kept identical; both omitted exportProject, so this case said nothing.
    render(
      <GlobalNotices
        connected
        mutations={[ok, ok, failed("Project export failed"), ok]}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Project export failed");
  });

  it("shows the first failure when several have failed", () => {
    render(
      <GlobalNotices connected mutations={[ok, failed("first"), failed("second")]} />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("first");
    expect(screen.queryByText("second")).not.toBeInTheDocument();
  });

  it("says nothing when everything succeeded and the socket is live", () => {
    render(<GlobalNotices connected mutations={[ok, ok]} />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("warns that a disconnected view may be out of date", () => {
    // refetchOnWindowFocus is off, so without this the app shows stale data
    // and looks perfectly healthy while doing it.
    render(<GlobalNotices connected={false} mutations={[ok]} />);

    expect(screen.getByRole("status")).toHaveTextContent(/out of date/i);
  });

  it("shows a failure and a dead socket at the same time", () => {
    render(<GlobalNotices connected={false} mutations={[failed("Send failed")]} />);

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Send failed");
  });

  it("shows a later failure after an earlier one was dismissed", () => {
    // The dismissed error stays on its mutation, and that mutation is first in
    // the list. Choosing the first failure and then hiding it if dismissed left
    // every later failure behind it invisible - a silent failure produced by
    // the control meant to make failures speak.
    const read = failed("Send failed");
    const { rerender } = render(<GlobalNotices connected mutations={[read, ok]} />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss error" }));

    rerender(<GlobalNotices connected mutations={[read, failed("Delete failed")]} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Delete failed");
  });

  it("keeps an earlier dismissal when a second failure is dismissed too", () => {
    const first = failed("Send failed");
    const second = failed("Delete failed");
    const { rerender } = render(<GlobalNotices connected mutations={[first, second]} />);

    fireEvent.click(screen.getByRole("button", { name: "Dismiss error" }));
    rerender(<GlobalNotices connected mutations={[first, second]} />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss error" }));
    rerender(<GlobalNotices connected mutations={[first, second]} />);

    // One slot for the dismissed error would have forgotten the first here.
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("dismisses a read failure and returns for a new one", () => {
    const first = failed("Send failed");
    const { rerender } = render(<GlobalNotices connected mutations={[first]} />);

    fireEvent.click(screen.getByRole("button", { name: "Dismiss error" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    // The same error instance stays dismissed across rerenders...
    rerender(<GlobalNotices connected mutations={[first]} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    // ...but a retried action that fails again is a new Error and must show.
    rerender(<GlobalNotices connected mutations={[failed("Send failed")]} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Send failed");
  });
});

describe("every mutation in the app", () => {
  it("leaves none with nowhere to report a failure", () => {
    // Centralizing the two `||` chains fixed this drift once; it then recurred
    // in nine places across seven files. Rating a response, cancelling a job,
    // favouriting a picture, archiving a family and asking for a diagnostics
    // bundle each changed a control that only moves on success, so a refusal
    // was indistinguishable on screen from a press that never happened.
    //
    // Three answers are honest: join the global list, hand the request to
    // FirstFailure, or report where the action lives. What is not an answer is
    // none of them, and reading twenty-odd names by eye never caught it.
    //
    // Every source file, not only the top-level components. This scan used to
    // read `.tsx` alone, and mutations declared in a `.ts` hook were invisible
    // to it - which is exactly where deleting a turn and forking a thread hid
    // with no failure surface at all, in the file this test was written to
    // protect. A hook's mutation is legitimately reported by whoever consumes
    // it, so reporting is collected across the whole app before judging.
    const sources = readdirSync(__dirname)
      .filter((file) => /\.tsx?$/.test(file) && !/\.test\.tsx?$/.test(file))
      .map((file) => ({ file, text: readFileSync(join(__dirname, file), "utf8") }));

    const app = sources.find(({ file }) => file === "App.tsx")!.text;
    const registration = /<GlobalNotices[^>]*mutations=\{\[([^\]]*)\]\}/.exec(app);
    expect(registration, "GlobalNotices is no longer given a list of mutations").toBeTruthy();

    const reported = new Set(registration![1].split(",").map((name) => name.trim()));
    for (const { text } of sources) {
      for (const list of text.matchAll(/<FirstFailure[^>]*of=\{\[([^\]]*)\]\}/g)) {
        for (const name of list[1].split(",")) reported.add(name.trim());
      }
    }

    const silent: string[] = [];
    for (const { file, text } of sources) {
      for (const match of text.matchAll(/const\s+(\w+)\s*=\s*useMutation[<(]/g)) {
        const name = match[1];
        if (reported.has(name)) continue;
        const declaration = text.slice(match.index, match.index + 1200);
        // Bounded on the left so `download.error` cannot vouch for a mutation
        // called `load`, and on the right so `keep.errorCount` cannot either.
        const reads = new RegExp(String.raw`(^|[^\w.])${name}\.(error|isError)\b`);
        const speaksAnywhere = sources.some(({ text: other }) => reads.test(other));
        if (!speaksAnywhere && !declaration.includes("onError")) silent.push(`${file}: ${name}`);
      }
    }

    expect(silent).toEqual([]);
  });
});
