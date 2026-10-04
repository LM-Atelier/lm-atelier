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
    updateWorkspaceLockPolicy: (write: WorkspaceLockPolicyWrite) =>
      request<WorkspaceLockPolicy>("/api/privacy/policy", { method: "PUT", body: JSON.stringify(write) }),
  };
}
