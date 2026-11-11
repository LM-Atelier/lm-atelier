import { useState } from "react";
import { Pencil, Trash2, X } from "lucide-react";

import { CopyTextButton } from "./CopyTextButton";
import { MessageTimestamp } from "./MessageTimestamp";

export function MessageRemovalConfirmation({
  messageId,
  onRemove,
  onKeep,
  disabled = false,
}: {
  messageId: string;
  onRemove: (messageId: string) => void;
  onKeep: () => void;
  disabled?: boolean;
}) {
  return (
    <span className="delete-confirm">
      <span>Only this item's content is removed. Replies stay.</span>
      <button className="danger" aria-disabled={disabled} onClick={() => { if (disabled) return; onRemove(messageId); }}>
        Remove this item, keep replies
      </button>
      <button onClick={onKeep}>Keep item</button>
    </span>
  );
}

export function UserMessageControls({
  messageId,
  createdAt,
  copyableText,
  onEdit,
  onDeleteExchange,
  onRemoveItem,
  disabled = false,
}: {
  messageId: string;
  createdAt: string;
  copyableText: string;
  onEdit?: () => void;
  onDeleteExchange?: (messageId: string) => void;
  onRemoveItem?: (messageId: string) => void;
  disabled?: boolean;
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [confirmingRemoval, setConfirmingRemoval] = useState(false);
  return (
    <div className="message-meta">
      <MessageTimestamp at={createdAt} />
      {confirmingRemoval && onRemoveItem ? (
        <MessageRemovalConfirmation
          messageId={messageId}
          disabled={disabled}
          onRemove={(id) => { setConfirmingRemoval(false); onRemoveItem(id); }}
          onKeep={() => setConfirmingRemoval(false)}
        />
      ) : confirmingDelete ? (
        <span className="delete-confirm">
          <span>Also deletes the answer and its media.</span>
          <button className="danger" aria-disabled={disabled} onClick={() => {
            if (disabled) return;
            setConfirmingDelete(false);
            onDeleteExchange?.(messageId);
          }}>Delete turn</button>
          <button onClick={() => setConfirmingDelete(false)}>Keep</button>
        </span>
      ) : (
        <span className="message-actions">
          {onEdit && <button onClick={onEdit} aria-label="Edit message" title="Edit"><Pencil size={14} /></button>}
          {copyableText && <CopyTextButton text={copyableText} label="Copy user message" buttonText="" />}
          {onRemoveItem && <button aria-disabled={disabled} aria-label="Remove this item, keep replies" title="Remove this item, keep replies" onClick={() => { if (disabled) return; setConfirmingRemoval(true); }}><X size={14} /></button>}
          {onDeleteExchange && <button aria-disabled={disabled} aria-label="Delete this turn" title="Delete turn" onClick={() => { if (disabled) return; setConfirmingDelete(true); }}><Trash2 size={14} /></button>}
        </span>
      )}
    </div>
  );
}
