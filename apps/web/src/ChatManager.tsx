import { useState } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { useConfirm } from "./useConfirm";
import type { Chat, Project } from "./types";

/** Rename, refile, archive, or delete one conversation. */
export function ChatManager({
  chat,
  projects,
  onClose,
  onSave,
  onDelete,
}: {
  chat: Chat;
  projects: Project[];
  onClose: () => void;
  onSave: (values: Partial<Chat>) => void;
  onDelete: (deleteGeneratedMedia: boolean) => void;
}) {
  const [confirmDialog, confirm] = useConfirm();
  const [title, setTitle] = useState(chat.title);
  const [projectId, setProjectId] = useState(chat.project_id ?? "");
  const [archived, setArchived] = useState(chat.archived);
  const [confirmUncertainMedia, setConfirmUncertainMedia] = useState(chat.confirm_uncertain_media);
  const [verifyImageEdits, setVerifyImageEdits] = useState(
    chat.vision_settings_json?.verify_image_edits === true,
  );
  const [compileVisualPrompts, setCompileVisualPrompts] = useState(
    chat.vision_settings_json?.compile_visual_prompts !== false,
  );
  const [deleteGeneratedMedia, setDeleteGeneratedMedia] = useState(false);
  const deletePrompt = deleteGeneratedMedia
    ? `Delete ${chat.title}, its history, and generated media used only by this chat?`
    : `Delete ${chat.title} and its history?`;
  return (
    <AccessibleDialog
      title="Manage chat"
      eyebrow="Conversation"
      closeLabel="Close chat manager"
      onClose={onClose}
      className="workspace-editor"
    >
      <label>Title<input value={title} onChange={(event) => setTitle(event.target.value)} /></label>
      <label>Project<select value={projectId} onChange={(event) => setProjectId(event.target.value)}><option value="">Unfiled</option>{projects.filter((project) => !project.archived).map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label>
      <label className="toggle-row"><span className="toggle-copy"><strong>Confirm uncertain media</strong><small>Ask before Auto mode starts an image or video when the planner is unsure.</small></span><input type="checkbox" checked={confirmUncertainMedia} onChange={(event) => setConfirmUncertainMedia(event.target.checked)} /></label>
      <label className="toggle-row"><span className="toggle-copy"><strong>Review image edits</strong><small>Check the result locally and retry once when the requested change is missing.</small></span><input type="checkbox" checked={verifyImageEdits} onChange={(event) => setVerifyImageEdits(event.target.checked)} /></label>
      <label className="toggle-row"><span className="toggle-copy"><strong>Compose visual prompts</strong><small>When a request asks to picture something written earlier, rewrite that passage as one scene description before generating.</small></span><input type="checkbox" checked={compileVisualPrompts} onChange={(event) => setCompileVisualPrompts(event.target.checked)} /></label>
      <label className="toggle-row"><span className="toggle-copy"><strong>Archived</strong><small>Hide this chat from the active workspace without deleting its history.</small></span><input type="checkbox" checked={archived} onChange={(event) => setArchived(event.target.checked)} /></label>
      <label className="toggle-row delete-media-option"><span className="toggle-copy"><strong>Delete generated media with chat</strong><small>Permanently delete image and video outputs used only by this chat. Shared media is kept.</small></span><input type="checkbox" checked={deleteGeneratedMedia} onChange={(event) => setDeleteGeneratedMedia(event.target.checked)} /></label>
      <footer className="editor-actions"><button className="secondary danger" onClick={() => void confirm({ title: "Delete this chat?", question: deletePrompt, confirmLabel: "Delete chat and history" }).then((ok) => ok && onDelete(deleteGeneratedMedia))}>Delete chat</button><button className="secondary" onClick={onClose}>Cancel</button><button className="primary" disabled={!title.trim()} onClick={() => onSave({ title: title.trim(), project_id: projectId || null, archived, confirm_uncertain_media: confirmUncertainMedia, vision_settings_json: { ...(chat.vision_settings_json ?? {}), verify_image_edits: verifyImageEdits, compile_visual_prompts: compileVisualPrompts } })}>Save chat</button></footer>
      {confirmDialog}
    </AccessibleDialog>
  );
}
