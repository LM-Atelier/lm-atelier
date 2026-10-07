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
    expect(link).toHaveAttribute("href", "/api/artifacts/sha256%3Aabc/export?format=jpeg&quality=90");

    fireEvent.change(screen.getByRole("combobox", { name: "Export format" }), { target: { value: "webp" } });
    expect(link).toHaveAttribute("href", "/api/artifacts/sha256%3Aabc/export?format=webp&quality=90");
  });

  it("offers a quality for JPEG and WebP only, and exports at the one chosen", () => {
    render(<StudioExportLink artifactId="sha256:abc" />);
    const link = screen.getByRole("link", { name: "Export" });
    const format = screen.getByRole("combobox", { name: "Export format" });
    expect(screen.queryByRole("combobox", { name: "Export quality" })).toBeNull();

    fireEvent.change(format, { target: { value: "jpeg" } });
    const quality = screen.getByRole("combobox", { name: "Export quality" });
    expect(quality).toHaveValue("90");
    fireEvent.change(quality, { target: { value: "75" } });
    expect(link).toHaveAttribute("href", "/api/artifacts/sha256%3Aabc/export?format=jpeg&quality=75");

    // The choice holds across the two lossy formats.
    fireEvent.change(format, { target: { value: "webp" } });
    expect(screen.getByRole("combobox", { name: "Export quality" })).toHaveValue("75");
    expect(link).toHaveAttribute("href", "/api/artifacts/sha256%3Aabc/export?format=webp&quality=75");

    // PNG loses nothing, so there is nothing to choose and nothing is sent.
    fireEvent.change(format, { target: { value: "png" } });
    expect(screen.queryByRole("combobox", { name: "Export quality" })).toBeNull();
    expect(link).toHaveAttribute("href", "/api/artifacts/sha256%3Aabc/export?format=png");
  });

  it("says that JPEG leaves out transparency when it is chosen", () => {
    render(<StudioExportLink artifactId="sha256:abc" />);
    const link = screen.getByRole("link", { name: "Export" });
    expect(link).not.toHaveAttribute("title");

    fireEvent.change(screen.getByRole("combobox", { name: "Export format" }), { target: { value: "jpeg" } });

    expect(link).toHaveAttribute("title", "JPEG has no transparency: transparent parts come out white.");
  });
});
