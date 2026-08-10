import { describe, expect, it } from "vitest";
import { catalogRoleFor, searchTermFor } from "./workflowAssetSearch";

describe("catalogRoleFor", () => {
  it("asks under a role the server accepts", () => {
    // `role` is chat, image or video. A LoRA was sent as "lora" and refused
    // with "Input should be 'chat', 'image' or 'video'", which reached the
    // user as a button that did nothing at all.
    expect(["chat", "image", "video"]).toContain(catalogRoleFor());
  });

  it("takes no asset kind, so a kind cannot leak into the role", () => {
    // lora-ness travels in auxiliary_kind, exactly as the model library
    // already does it. The parameter is gone so the bug cannot come back by
    // someone special-casing a kind here again.
    expect(catalogRoleFor.length).toBe(0);
    expect(catalogRoleFor()).toBe("image");
  });
});

describe("searchTermFor", () => {
  it("turns a filename into words a catalog can match", () => {
    expect(searchTermFor("wan_2.1_vae.safetensors")).toBe("wan 2.1 vae");
  });

  it("drops the directories a workflow wrote around the file", () => {
    // A repository file is not indexed by its path, so searching for one
    // finds nothing.
    expect(searchTermFor("split_files/vae/wan_2.1_vae.safetensors")).toBe("wan 2.1 vae");
  });

  it("handles a windows separator the same way", () => {
    expect(searchTermFor("split_files\\vae\\wan_2.1_vae.safetensors")).toBe("wan 2.1 vae");
  });
});
