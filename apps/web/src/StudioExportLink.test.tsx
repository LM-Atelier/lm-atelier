/** Exporting from the studio: the file as stored, or made in another format on the way out. */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { StudioExportLink } from "./StudioExportLink";

afterEach(cleanup);

describe("the export control", () => {
  it("downloads the file as stored until another format is chosen", () => {
    render(<StudioExportLink artifactId="sha256:abc" />);
    const link = screen.getByRole("link", { name: "Export" });

    expect(link).toHaveAttribute("download");
    expect(link).toHaveAttribute("href", "/api/artifacts/sha256%3Aabc/content");

    fireEvent.change(screen.getByRole("combobox", { name: "Export format" }), { target: { value: "jpeg" } });
    expect(link).toHaveAttribute("href", "/api/artifacts/sha256%3Aabc/export?format=jpeg");

    fireEvent.change(screen.getByRole("combobox", { name: "Export format" }), { target: { value: "webp" } });
    expect(link).toHaveAttribute("href", "/api/artifacts/sha256%3Aabc/export?format=webp");
  });

  it("says that JPEG leaves out transparency when it is chosen", () => {
    render(<StudioExportLink artifactId="sha256:abc" />);
    const link = screen.getByRole("link", { name: "Export" });
    expect(link).not.toHaveAttribute("title");

    fireEvent.change(screen.getByRole("combobox", { name: "Export format" }), { target: { value: "jpeg" } });

    expect(link).toHaveAttribute("title", "JPEG has no transparency: transparent parts come out white.");
  });
});
