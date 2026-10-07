export interface CustomNodeContainmentStatus {
  level:
    | "unavailable"
    | "process_tree_only"
    | "filesystem_restricted"
    | "filesystem_no_egress"
    | "verified";
  platform: string;
  profile_version: number;
  backend: string;
  backend_version: string;
  profile_sha256?: string | null;
  file_denial_provable: boolean;
  connect_denial_provable: boolean;
  authorizes_execution: boolean;
  offline_badge: boolean;
}
