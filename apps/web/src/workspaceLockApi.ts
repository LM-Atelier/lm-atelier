import type { WorkspaceLockPolicy, WorkspaceLockPolicyWrite, WorkspaceLockStatus } from "./workspaceLockTypes";

type Request = <T>(path: string, init?: RequestInit) => Promise<T>;

/** The workspace lock's requests. A PIN only ever travels in a request body. */
export function workspaceLockApi(request: Request) {
  return {
    workspaceLockStatus: () => request<WorkspaceLockStatus>("/api/privacy/status"),
    lockWorkspace: () => request<WorkspaceLockStatus>("/api/privacy/lock", { method: "POST" }),
    unlockWorkspace: (pin?: string) =>
      request<WorkspaceLockStatus>("/api/privacy/unlock", {
        method: "POST",
        body: JSON.stringify(pin === undefined ? {} : { pin }),
      }),
    workspaceLockPolicy: () => request<WorkspaceLockPolicy>("/api/privacy/policy"),
    // Only a key, a click or a touch is reported, and the server lets none of
    // these through while locked, so this can never lift a lock.
    noteWorkspaceUse: () => request<void>("/api/privacy/activity", { method: "POST" }),
    updateWorkspaceLockPolicy: (write: WorkspaceLockPolicyWrite) =>
      request<WorkspaceLockPolicy>("/api/privacy/policy", { method: "PUT", body: JSON.stringify(write) }),
  };
}
