import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { RecoveryBatchAction, RecoveryBatchCommand, RecoveryBatchPreview, RecoveryItem } from "./recoveryTypes";

type Intent = { items: RecoveryItem[]; action: RecoveryBatchAction; restoreUnfiled: boolean; operationKey: string };
type Frozen = { preview: RecoveryBatchPreview; command: RecoveryBatchCommand };

export function useRecoveryBatch() {
  const client = useQueryClient();
  const [selected, setSelected] = useState<RecoveryItem[]>([]);
  const [intent, setIntent] = useState<Intent | null>(null);
  const [open, setOpen] = useState(false);
  const [frozen, setFrozen] = useState<Frozen | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const submitting = useRef(false);
  const preview = useQuery({
    queryKey: ["recovery-batch-preview", intent?.operationKey],
    queryFn: async ({ signal }) => {
      const value = await api.previewRecoveryBatch({
        deletion_ids: intent!.items.map(item => item.deletion_id), action: intent!.action,
        restore_unfiled: intent!.restoreUnfiled,
      }, signal);
      return { value, receivedAt: Date.now() };
    },
    enabled: intent !== null,
    staleTime: Infinity,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
  useEffect(() => {
    if (!preview.data) return;
    const delay = new Date(preview.data.value.expires_at).getTime() - Date.now();
    const timer = window.setTimeout(() => setNow(Date.now()), Math.max(0, Math.min(delay, 2_147_483_647)));
    return () => window.clearTimeout(timer);
  }, [preview.data]);
  const transition = useMutation({
    mutationFn: (request: Frozen) => api.applyRecoveryBatch(request.preview.batch_id, request.command),
    onSuccess: (result) => {
      const ids = new Set(result.results.map(item => item.deletion_id));
      setSelected(current => current.filter(item => !ids.has(item.deletion_id)));
      setNotice(result.action === "restore"
        ? `${result.results.length} items restored together. No work restarted; workflows remain disabled.`
        : `${result.results.length} items permanently deleted together. Shared and retained media remains available.`);
      setIntent(null); setOpen(false); setFrozen(null); setAcknowledged(false);
      for (const key of ["recovery-items", "chats", "chat-summaries", "chat", "chat-management", "projects", "empty-chats",
        "storage", "artifact-library-v1", "artifacts", "artifact-storage", "workflow-families", "workflow-family",
        "workflows", "workflow-revision", "workflow-ready-revisions", "studio-capabilities"])
        void client.invalidateQueries({ queryKey: [key] });
    },
    onSettled: () => { submitting.current = false; },
  });
  const busy = transition.isPending;
  const rejected = transition.error instanceof ApiError && [404, 409, 410, 422].includes(transition.error.status);
  const displayedPreview = frozen?.preview ?? preview.data?.value;
  const expired = displayedPreview ? new Date(displayedPreview.expires_at).getTime() <= Math.max(now, preview.data?.receivedAt ?? 0) : false;
  const canConfirm = Boolean(intent && displayedPreview?.available && (frozen || (!preview.isFetching && !preview.isError))
    && !busy && !rejected && (!expired || frozen) && (intent.action !== "purge" || acknowledged));

  function toggle(item: RecoveryItem) {
    if (intent || submitting.current || item.state !== "recoverable") return;
    setSelected(current => current.some(value => value.deletion_id === item.deletion_id)
      ? current.filter(value => value.deletion_id !== item.deletion_id)
      : current.length < 20 ? [...current, item] : current);
  }
  function clear() { if (!intent && !submitting.current) setSelected([]); }
  function review(action: RecoveryBatchAction) {
    if (intent || submitting.current || !selected.length) return;
    transition.reset(); setFrozen(null); setAcknowledged(false); setNotice(null); setNow(Date.now());
    setIntent({ items: [...selected], action, restoreUnfiled: false, operationKey: crypto.randomUUID() });
    setOpen(true);
  }
  function recheck(restoreUnfiled = intent?.restoreUnfiled ?? false) {
    if (!intent || submitting.current || busy || (frozen && !rejected)) return;
    transition.reset(); setFrozen(null); setAcknowledged(false); setNow(Date.now());
    setIntent({ ...intent, restoreUnfiled, operationKey: crypto.randomUUID() });
  }
  function confirm() {
    if (!canConfirm || !intent || !displayedPreview || submitting.current) return;
    const request = frozen ?? { preview: displayedPreview, command: {
      expected_revision: displayedPreview.revision, impact_sha256: displayedPreview.impact_sha256, operation_key: intent.operationKey,
      ...(intent.action === "purge" ? { acknowledgement: "permanently-delete" as const } : {}),
    } };
    submitting.current = true; setFrozen(request); transition.mutate(request);
  }
  function dismiss() {
    if (busy || submitting.current) return;
    setOpen(false);
    if (!frozen || rejected) { setIntent(null); setFrozen(null); transition.reset(); }
  }
  return {
    selected, intent, open, frozen, acknowledged, notice, preview, displayedPreview, transition, busy, rejected, expired, canConfirm,
    toggle, clear, review, recheck, confirm, dismiss,
    reopen: () => setOpen(true),
    acknowledge: (value: boolean) => { if (!busy && !frozen) setAcknowledged(value); },
  };
}
