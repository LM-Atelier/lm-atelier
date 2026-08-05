import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { CatalogVariantChoice } from "./CatalogVariantChoice";
import type { CatalogFileVariant } from "./types";

function variant(id: string, precision: string | null, size: number | null): CatalogFileVariant {
  return { source_file_id: id, filename: "lustify.safetensors", size_bytes: size, precision };
}

describe("CatalogVariantChoice", () => {
  afterEach(cleanup);

  it("offers each variant with what tells them apart", () => {
    const onChoose = vi.fn();
    render(
      <CatalogVariantChoice
        filename="lustify.safetensors"
        variants={[variant("301", "fp16", 6_400_000_000), variant("302", "fp8", 3_200_000_000)]}
        onChoose={onChoose}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /fp8/ }));

    // The immutable id, never the filename that could not settle it.
    expect(onChoose).toHaveBeenCalledWith("302");
    expect(screen.getByRole("button", { name: /fp16 · 6.4 GB/ })).toBeInTheDocument();
  });

  it("says nothing when there is nothing to choose between", () => {
    // A list of one would turn an ordinary install into a decision.
    const { container } = render(
      <CatalogVariantChoice
        filename="one.safetensors"
        variants={[variant("301", "fp16", 1)]}
        onChoose={vi.fn()}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("names a variant the provider left unlabelled rather than showing a blank", () => {
    render(
      <CatalogVariantChoice
        filename="lustify.safetensors"
        variants={[variant("301", null, null), variant("302", "fp8", null)]}
        onChoose={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: /^Unlabelled · #/ })).toBeInTheDocument();
  });

  it("tells apart two variants that differ in nothing a person can see", () => {
    // The live case: two rows sharing a precision and an exact byte count.
    // Without the identity in the label both buttons carry the same name and
    // neither can be pointed at.
    const onChoose = vi.fn();
    render(
      <CatalogVariantChoice
        filename="lustify.safetensors"
        variants={[
          variant("2996235", "bf16", 13_148_974_712),
          variant("2997070", "bf16", 13_148_974_712),
        ]}
        onChoose={onChoose}
      />,
    );

    const names = screen.getAllByRole("button").map((button) => button.textContent);
    expect(new Set(names).size).toBe(2);

    fireEvent.click(screen.getByRole("button", { name: /997070/ }));

    expect(onChoose).toHaveBeenCalledWith("2997070");
  });
});
