import { useRef, useState } from "react";
import { api, ApiError } from "./api";
import { organizationOperationKey, parseMediaAlbum, parseMediaTag } from "./mediaOrganization";

type Creation = { kind: "albums" | "tags"; name: string; operationKey: string };

export function useMediaOrganizationCreation(onCreated: () => void) {
  const [request, setRequest] = useState<Creation | null>(null);
  const held = useRef<Creation | null>(null);
  const submitting = useRef(false);
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState(false);
  const create = async (kind: Creation["kind"], name: string): Promise<boolean> => {
    if (submitting.current || !name.trim()) return false;
    const old = held.current;
    if (old && (old.kind !== kind || old.name !== name.trim())) { setFailed(true); return false; }
    const choice = old ?? { kind, name: name.trim(), operationKey: organizationOperationKey() };
    held.current = choice; setRequest(choice);
    submitting.current = true; setPending(true); setFailed(false);
    try {
      if (kind === "albums") {
        const result = parseMediaAlbum(await api.createMediaCollection(choice.name, "", choice.operationKey));
        if (result.name !== choice.name || result.description !== "" || result.version !== 1) throw new Error("creation-response-invalid");
      } else {
        const result = parseMediaTag(await api.createMediaTag(choice.name, null, choice.operationKey));
        if (result.label !== choice.name || result.color !== null || result.version !== 1) throw new Error("creation-response-invalid");
      }
      held.current = null; setRequest(null); onCreated();
      return true;
    } catch (error) {
      const refused = error instanceof ApiError && (
        (error.status === 422 && error.code === (kind === "albums" ? "media-collection-invalid" : "media-tag-invalid"))
        || (kind === "tags" && error.status === 409 && error.code === "media-tag-conflict")
      );
      if (refused) { held.current = null; setRequest(null); }
      setFailed(true); return false;
    }
    finally { submitting.current = false; setPending(false); }
  };
  return { request, pending, failed, create };
}
