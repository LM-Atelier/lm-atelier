import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { clockOptions, useClockChoice } from "./clockPreference";
import { ApiError, api } from "./api";
import { adjacentMove, droppedMove, orderOwnerKey, QUEUE_ORDER_REASONS } from "./queueOrderMoves";
import type { QueueLane, QueueOrderCommand, QueueOrderItem, QueueOrderPage } from "./queueOrderTypes";
import "./QueueOrderPanel.css";
import "./QueueActivityDialog.css";

export function QueueOrderPanel({ initialLane, onBack }: { initialLane: QueueLane; onBack: () => void }) {
  const [lane, setLane] = useState(initialLane);
  return (
    <section className="queue-order-panel" aria-label="Dispatch order">
      <div className="queue-activity-toolbar">
        <label>Order category
          <select value={lane} onChange={(event) => setLane(event.target.value as QueueLane)}>
            <option value="generation">Generation</option>
            <option value="transfer">Transfers</option>
            <option value="install">Installs</option>
          </select>
        </label>
        <button className="secondary compact-button" onClick={onBack}>Back to accepted work</button>
      </div>
      <p>Move work only while this category is idle. Each group shares a resource and priority.
        Older work keeps its priority as it waits; these positions are within a group.
        Use Move earlier or Move later, or drag to the top or bottom of another item.</p>
      <QueueOrderLane key={lane} lane={lane} />
    </section>
  );
}

function QueueOrderLane({ lane }: { lane: QueueLane }) {
  const client = useQueryClient();
  const clock = useClockChoice();
  const [cursors, setCursors] = useState<Array<string | null>>([null]);
  const cursor = cursors[cursors.length - 1];
  const queryKey = ["jobs", "queue", "order", lane] as const;
  const page = useQuery({
    queryKey: [...queryKey, cursor],
    queryFn: ({ signal }) => api.queueOrder(lane, { cursor, limit: 50 }, signal),
    retry: false,
    refetchInterval: 5_000,
  });
  const [attempt, setAttempt] = useState<QueueOrderCommand | null>(null);
  const [notice, setNotice] = useState("");
  const [dropAt, setDropAt] = useState<{ key: string; direction: "before" | "after" } | null>(null);
  const sending = useRef(false);
  const drag = useRef<{ page: QueueOrderPage; item: QueueOrderItem } | null>(null);
  const focused = useRef<HTMLElement | null>(null);
  const refreshButton = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    if (focused.current && !focused.current.isConnected && document.activeElement === document.body) {
      refreshButton.current?.focus();
    }
  }, [page.data]);
  const refresh = async () => {
    setCursors([null]);
    await client.invalidateQueries({ queryKey: ["jobs", "queue"] });
  };
  const mutation = useMutation({
    mutationFn: (command: QueueOrderCommand) => api.reorderQueue(lane, command),
    onSuccess: async () => {
      setAttempt(null);
      setNotice("Order saved. The list now shows the latest positions.");
      await refresh();
    },
    onError: async (error) => {
      if (error instanceof ApiError && error.status === 409) {
        setAttempt(null);
        setNotice(error.code === "queue-order-limit-exceeded"
          ? error.message : "The queue changed. Review its latest order before moving work again.");
        await refresh();
      }
    },
    onSettled: () => { sending.current = false; },
  });
  const unavailable = page.isError || mutation.isPending || attempt !== null;
  const submit = (command: QueueOrderCommand | null) => {
    if (!command || sending.current) return;
    sending.current = true;
    setNotice("");
    setAttempt(command);
    mutation.mutate(command);
  };
  const move = (item: QueueOrderItem, direction: "before" | "after") => {
    if (!page.data || unavailable) return;
    submit(adjacentMove(page.data, item, direction, crypto.randomUUID()));
  };
  const conflict = mutation.error instanceof ApiError && mutation.error.status === 409;
  return (
    <div aria-busy={mutation.isPending}
      onFocusCapture={(event) => { focused.current = event.target; }}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) focused.current = null;
      }}>
      <div className="queue-activity-toolbar">
        <button ref={refreshButton} className="secondary compact-button" aria-disabled={page.isFetching || mutation.isPending}
          onClick={() => { if (!page.isFetching && !mutation.isPending) void refresh(); }}>
          Refresh dispatch order
        </button>
        {page.data && <small>{page.data.total} items in this category</small>}
      </div>
      <p role="status" aria-live="polite">{mutation.isPending ? "Saving order..." : notice}</p>
      {page.isPending && <p role="status">Loading dispatch order...</p>}
      {page.error && <p role="alert">{page.error instanceof ApiError && page.error.status === 409
        && page.error.code !== "queue-order-limit-exceeded"
        ? "This page changed. Refresh dispatch order to load it again." : page.error.message}</p>}
      {mutation.error && !conflict && <div className="queue-activity-error">
        <p role="alert">{mutation.error.message}</p>
        {attempt && <button className="secondary compact-button" aria-disabled={mutation.isPending}
          onClick={() => { if (!mutation.isPending) submit(attempt); }}>Retry the same move</button>}
      </div>}
      {page.data && !page.data.items.length && <p>No work to order in this category.</p>}
      <ol className="queue-activity-items queue-order-items" aria-label="Work in dispatch groups">
        {page.data?.items.map((item) => {
          const canMove = !item.unavailable_reason && item.cohort_id !== null && item.cohort_length > 1;
          return (
            <li key={orderOwnerKey(item)} aria-label={item.label}
              data-drop-placement={dropAt?.key === orderOwnerKey(item) ? dropAt.direction : undefined}
              onDragLeave={() => setDropAt(null)} onDragOver={(event) => {
              if (!unavailable && canMove && drag.current?.item.cohort_id === item.cohort_id) {
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
                const bounds = event.currentTarget.getBoundingClientRect();
                setDropAt({ key: orderOwnerKey(item),
                  direction: event.clientY < bounds.top + bounds.height / 2 ? "before" : "after" });
              }
            }} onDrop={(event) => {
              event.preventDefault();
              setDropAt(null);
              const started = drag.current;
              drag.current = null;
              if (!started || !page.data || unavailable || !canMove) return;
              const current = page.data.items.find((row) => orderOwnerKey(row) === orderOwnerKey(started.item));
              if (!current || started.page.revision !== page.data.revision
                || started.item.cohort_id !== current.cohort_id) {
                setNotice("The queue changed during the drag. Review its latest order.");
                void refresh();
                return;
              }
              const bounds = event.currentTarget.getBoundingClientRect();
              submit(droppedMove(page.data, current, item,
                event.clientY < bounds.top + bounds.height / 2 ? "before" : "after", crypto.randomUUID()));
            }}>
              <strong>{item.label}</strong>
              <small>Queued {new Date(item.queued_at).toLocaleString(undefined, clockOptions(clock))}</small>
              {item.priority !== null && <small>Priority {item.priority}</small>}
              {item.position !== null && <small>Position {item.position} of {item.cohort_length} in its group</small>}
              {item.unavailable_reason && <p>{QUEUE_ORDER_REASONS[item.unavailable_reason]}</p>}
              {!item.unavailable_reason && item.cohort_length === 1 && <p>No other work shares this group.</p>}
              {canMove && <div className="queue-order-actions">
                <button className="secondary compact-button" draggable={!unavailable}
                  aria-disabled={unavailable} aria-label={"Drag " + item.label}
                  onDragStart={(event) => {
                    if (unavailable || !page.data) { event.preventDefault(); return; }
                    drag.current = { page: page.data, item };
                    event.dataTransfer.effectAllowed = "move";
                    event.dataTransfer.setData("text/plain", "queue-item");
                  }} onDragEnd={() => { drag.current = null; setDropAt(null); }}>Drag to move</button>
                <button className="secondary compact-button" aria-disabled={unavailable || !item.neighbors.before}
                  onClick={() => move(item, "before")}>Move earlier</button>
                <button className="secondary compact-button" aria-disabled={unavailable || !item.neighbors.after}
                  onClick={() => move(item, "after")}>Move later</button>
              </div>}
            </li>
          );
        })}
      </ol>
      <div className="queue-order-actions">
        {cursors.length > 1 && <button className="secondary compact-button" aria-disabled={unavailable}
          onClick={() => { if (!unavailable) setCursors((current) => current.slice(0, -1)); }}>Previous order page</button>}
        {page.data?.next_cursor && <button className="secondary compact-button" aria-disabled={unavailable}
          onClick={() => {
            if (!unavailable && page.data?.next_cursor) setCursors((current) => [...current, page.data.next_cursor]);
          }}>Next order page</button>}
      </div>
    </div>
  );
}
