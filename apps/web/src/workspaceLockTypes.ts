/** Whether the workspace is locked, as the server reports it. */
export interface WorkspaceLockStatus {
  locked: boolean;
  enabled: boolean;
  require_pin: boolean;
  /** Changes every time the workspace locks and every time the service starts; null while the lock is off. */
  lock_epoch: string | null;
}

/** The saved lock setting. Whether a PIN is set is all it says about the PIN. */
export interface WorkspaceLockPolicy {
  enabled: boolean;
  require_pin: boolean;
  revision: number;
}

/** A change to the lock setting, made against the revision it was read at. */
export interface WorkspaceLockPolicyWrite {
  expected_revision: number;
  enabled: boolean;
  new_pin?: string | null;
  clear_pin?: boolean;
  current_pin?: string | null;
}
