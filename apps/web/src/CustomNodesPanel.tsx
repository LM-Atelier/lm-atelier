import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ConfirmDialog, PromptDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import type { CustomNodeInstall } from "./types";

function reviewedNodeTypes(node: CustomNodeInstall): string[] {
  const value = node.security_json.node_types;
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    ? value
    : [];
}

function parseNodeTypes(value: string): string[] {
  return [...new Set(value.split(/[\n,]/).map((item) => item.trim()).filter(Boolean))]
    .sort((left, right) => left.localeCompare(right));
}

export function CustomNodesPanel() {
  const client = useQueryClient();
  const nodes = useQuery({ queryKey: ["custom-nodes"], queryFn: api.customNodes });
  const [name, setName] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [revision, setRevision] = useState("");
  const refresh = () => void client.invalidateQueries({ queryKey: ["custom-nodes"] });
  const install = useMutation({ mutationFn: () => api.installCustomNode({ name: name.trim(), source_url: sourceUrl.trim(), revision: revision.trim() }), onSuccess: () => { setName(""); setSourceUrl(""); setRevision(""); refresh(); } });
  const trust = useMutation({
    mutationFn: ({ id, trusted, nodeTypes = [] }: {
      id: string;
      trusted: boolean;
      nodeTypes?: string[];
    }) => api.trustCustomNode(id, trusted, nodeTypes),
    onSuccess: refresh,
  });
  const update = useMutation({ mutationFn: ({ id, revision: next }: { id: string; revision: string }) => api.updateCustomNode(id, next), onSuccess: refresh });
  const rollback = useMutation({ mutationFn: api.rollbackCustomNode, onSuccess: refresh });
  const remove = useMutation({ mutationFn: api.removeCustomNode, onSuccess: refresh });
  const error = install.error || trust.error || update.error || rollback.error || remove.error;
  const [asking, setAsking] = useState<{ kind: "install" | "trust" | "remove" | "revision"; node?: CustomNodeInstall } | null>(null);
  const [nodeTypes, setNodeTypes] = useState("");
  const close = () => setAsking(null);
  const reviewTrust = (node: CustomNodeInstall) => {
    setNodeTypes(reviewedNodeTypes(node).join("\n"));
    setAsking({ kind: "trust", node });
  };
  return <section className="custom-nodes"><div className="detail-title"><div><h2>Custom nodes</h2><p>Pinned sources stay disabled until you review and trust the exact revision. Stop ComfyUI before changing nodes.</p></div></div><div className="custom-node-install"><input aria-label="Custom node name" placeholder="Node name" value={name} onChange={(event) => setName(event.target.value)} /><input aria-label="Custom node source" placeholder="https://github.com/owner/repository" value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} /><input aria-label="Custom node commit" placeholder="Full 40-character commit SHA" value={revision} onChange={(event) => setRevision(event.target.value)} /><button className="primary" disabled={!name.trim() || !sourceUrl.trim() || revision.trim().length !== 40 || install.isPending} onClick={() => setAsking({ kind: "install" })}>Install pinned source</button></div>{error && <ErrorCallout message={error.message} />}<div className="profile-table custom-node-list">{nodes.data?.map((node) => <div key={node.id}><span className={`badge ${node.trusted ? "likely" : "advanced_import"}`}>{node.trusted ? "Trusted" : "Review required"}</span><span><strong>{node.name}</strong><small>{node.source_url}<br />{node.revision}</small></span><details><summary>Security</summary><pre>{JSON.stringify(node.security_json, null, 2)}</pre></details><span className="row-actions"><button className="secondary compact-button" onClick={() => setAsking({ kind: "revision", node })}>Update</button>{node.previous_revision && <button className="secondary compact-button" onClick={() => rollback.mutate(node.id)}>Rollback</button>}<button className="secondary compact-button" onClick={() => node.trusted ? trust.mutate({ id: node.id, trusted: false }) : reviewTrust(node)}>{node.trusted ? "Revoke trust" : "Trust revision"}</button><button className="secondary compact-button danger" onClick={() => setAsking({ kind: "remove", node })}>Remove</button></span></div>)}</div>
{asking?.kind === "install" && <ConfirmDialog title="Download this pinned repository?" question="Its code stays untrusted until you review this exact revision and say so. Downloading alone does not run anything." confirmLabel="Download for review" tone="trust" onCancel={close} onConfirm={() => { close(); install.mutate(); }} />}
{asking?.kind === "trust" && asking.node && <ConfirmDialog title={`Trust ${asking.node.name}?`} question="Trusting this revision lets its code run inside ComfyUI on this machine. Confirm only if you have read this exact pinned revision." detail={<><dl className="confirm-facts"><div><dt>Source</dt><dd>{asking.node.source_url}</dd></div><div><dt>Pinned revision</dt><dd><code>{asking.node.revision}</code></dd></div></dl><label>Reviewed node types<textarea value={nodeTypes} onChange={(event) => setNodeTypes(event.target.value)} placeholder={"One exact ComfyUI node type per line\nLeave blank if you have not reviewed its node inventory"} /></label><p className="package-review-note">Only names you confirm here can satisfy a workflow package or enter an activation scope.</p></>} confirmLabel="I reviewed this revision - trust it" tone="trust" onCancel={close} onConfirm={() => { const node = asking.node!; const reviewed = parseNodeTypes(nodeTypes); close(); trust.mutate({ id: node.id, trusted: true, nodeTypes: reviewed }); }} />}
{asking?.kind === "remove" && asking.node && <ConfirmDialog title={`Remove ${asking.node.name}?`} question="This deletes the downloaded repository. Any workflow depending on its nodes stops resolving until it is installed again." confirmLabel="Remove" onCancel={close} onConfirm={() => { const node = asking.node!; close(); remove.mutate(node.id); }} />}
{asking?.kind === "revision" && asking.node && <PromptDialog title={`Update ${asking.node.name}`} label="Full pinned commit SHA" initialValue={asking.node.revision} placeholder="40 hexadecimal characters" confirmLabel="Pin this revision" validate={(value) => { const next = value.trim(); if (!next) return null; if (!/^[0-9a-f]{40}$/i.test(next)) return "A pinned revision is a full 40-character commit SHA."; if (next === asking.node!.revision) return "That is the revision already pinned."; return null; }} onCancel={close} onConfirm={(next) => { const node = asking.node!; close(); update.mutate({ id: node.id, revision: next }); }} />}
</section>;
}
