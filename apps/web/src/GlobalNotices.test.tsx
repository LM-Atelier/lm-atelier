import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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
