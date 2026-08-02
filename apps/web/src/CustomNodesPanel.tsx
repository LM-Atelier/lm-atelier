import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";

export function CustomNodesPanel() {
  const client = useQueryClient();
  const nodes = useQuery({ queryKey: ["custom-nodes"], queryFn: api.customNodes });
  const [name, setName] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [revision, setRevision] = useState("");
  const refresh = () => void client.invalidateQueries({ queryKey: ["custom-nodes"] });
  const install = useMutation({ mutationFn: () => api.installCustomNode({ name: name.trim(), source_url: sourceUrl.trim(), revision: revision.trim() }), onSuccess: () => { setName(""); setSourceUrl(""); setRevision(""); refresh(); } });
  const trust = useMutation({ mutationFn: ({ id, trusted }: { id: string; trusted: boolean }) => api.trustCustomNode(id, trusted), onSuccess: refresh });
  const update = useMutation({ mutationFn: ({ id, revision: next }: { id: string; revision: string }) => api.updateCustomNode(id, next), onSuccess: refresh });
  const rollback = useMutation({ mutationFn: api.rollbackCustomNode, onSuccess: refresh });
  const remove = useMutation({ mutationFn: api.removeCustomNode, onSuccess: refresh });
  const error = install.error || trust.error || update.error || rollback.error || remove.error;
  return <section className="custom-nodes"><div className="detail-title"><div><h2>Custom nodes</h2><p>Pinned sources stay disabled until you review and trust the exact revision. Stop ComfyUI before changing nodes.</p></div></div><div className="custom-node-install"><input aria-label="Custom node name" placeholder="Node name" value={name} onChange={(event) => setName(event.target.value)} /><input aria-label="Custom node source" placeholder="https://github.com/owner/repository" value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} /><input aria-label="Custom node commit" placeholder="Full 40-character commit SHA" value={revision} onChange={(event) => setRevision(event.target.value)} /><button className="primary" disabled={!name.trim() || !sourceUrl.trim() || revision.trim().length !== 40 || install.isPending} onClick={() => { if (window.confirm("Download this pinned repository for review? Its code will remain untrusted.")) install.mutate(); }}>Install pinned source</button></div>{error && <ErrorCallout message={error.message} />}<div className="profile-table custom-node-list">{nodes.data?.map((node) => <div key={node.id}><span className={`badge ${node.trusted ? "likely" : "advanced_import"}`}>{node.trusted ? "Trusted" : "Review required"}</span><span><strong>{node.name}</strong><small>{node.source_url}<br />{node.revision}</small></span><details><summary>Security</summary><pre>{JSON.stringify(node.security_json, null, 2)}</pre></details><span className="row-actions"><button className="secondary compact-button" onClick={() => { const next = window.prompt("Full pinned commit SHA", node.revision); if (next?.trim() && next.trim() !== node.revision) update.mutate({ id: node.id, revision: next.trim() }); }}>Update</button>{node.previous_revision && <button className="secondary compact-button" onClick={() => rollback.mutate(node.id)}>Rollback</button>}<button className="secondary compact-button" onClick={() => node.trusted ? trust.mutate({ id: node.id, trusted: false }) : window.confirm("I reviewed this exact pinned revision and trust its code to run in ComfyUI.") && trust.mutate({ id: node.id, trusted: true })}>{node.trusted ? "Revoke trust" : "Trust revision"}</button><button className="secondary compact-button danger" onClick={() => window.confirm(`Remove ${node.name}?`) && remove.mutate(node.id)}>Remove</button></span></div>)}</div></section>;
}
