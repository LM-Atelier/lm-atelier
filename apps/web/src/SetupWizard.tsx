import { useEffect, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Bot, Check, Film, Image as ImageIcon, LoaderCircle } from "lucide-react";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { api } from "./api";
import { formatBytes } from "./format";
import { jobProgressFraction, jobProgressText } from "./jobProgress";
import type {
  ReferenceRecipe,
  RuntimeStatus,
  SetupReadinessReport,
  SetupRoleReadiness,
  SystemInfo,
} from "./types";

function setupRoleName(role: SetupRoleReadiness["role"]): string {
  return role === "chat" ? "Chat" : role === "image" ? "Image" : "Video";
}

function recipeFitsSystem(recipe: ReferenceRecipe, system: SystemInfo): boolean {
  if (system.memory_total_bytes < recipe.hardware.minimum_ram_gb * 1024 ** 3) return false;
  if (recipe.total_size_bytes != null && system.disk_free_bytes < recipe.total_size_bytes) return false;
  if (recipe.hardware.minimum_vram_gb == null) return true;
  return system.devices.some(
    (device) =>
      device.total_memory_bytes != null
      && device.total_memory_bytes >= recipe.hardware.minimum_vram_gb! * 1024 ** 3,
  );
}

export function SetupWizard({
  report,
  onClose,
  onOpenModels,
  onOpenWorkflows,
  autoPrepare = false,
}: {
  report: SetupReadinessReport;
  onClose: () => void;
  onOpenModels: (role: SetupRoleReadiness["role"]) => void;
  onOpenWorkflows: () => void;
  // First-run installs prepare workers without being asked: the whole point
  // of installer-time setup is that the first real request pays nothing.
  autoPrepare?: boolean;
}) {
  const client = useQueryClient();
  const recipes = useQuery({ queryKey: ["recipes"], queryFn: api.recipes });
  const system = useQuery({ queryKey: ["system"], queryFn: api.system });
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: report.state === "ready" ? false : 3_000,
  });
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["setup-readiness"] });
    void client.invalidateQueries({ queryKey: ["jobs"] });
    void client.invalidateQueries({ queryKey: ["models"] });
    void client.invalidateQueries({ queryKey: ["profiles"] });
    void client.invalidateQueries({ queryKey: ["runtimes"] });
    void client.invalidateQueries({ queryKey: ["workers"] });
    void client.invalidateQueries({ queryKey: ["workflows"] });
  };
  const installRecipe = useMutation({
    mutationFn: (recipeId: string) => api.installRecipe(recipeId),
    onSuccess: refresh,
  });
  const retryInstall = useMutation({
    mutationFn: (jobId: string) => api.resumeDownload(jobId),
    onSuccess: refresh,
  });
  const activateModel = useMutation({
    mutationFn: (installId: string) => api.activateModel(installId),
    onSuccess: refresh,
  });
  const installRuntime = useMutation({
    mutationFn: (engine: RuntimeStatus["engine"]) => api.installRuntime(engine),
    onSuccess: refresh,
  });
  // Serves both repair and preparation: restarting a failed worker and loading
  // a working one ahead of the first request are the same call.
  const startWorker = useMutation({
    mutationFn: (role: SetupRoleReadiness) => {
      if (role.role === "chat" && role.profile_id) return api.loadChatWorker(role.profile_id);
      if (role.role !== "chat") return api.startMediaWorker();
      throw new Error("Choose a chat model before starting its worker.");
    },
    onSuccess: refresh,
  });
  const verifyRole = useMutation({
    mutationFn: (role: SetupRoleReadiness["role"]) => api.verifySetupRole(role),
    onSuccess: refresh,
  });
  const suitableRecipes = (role: SetupRoleReadiness["role"]) => (
    !system.data
      ? []
      : (recipes.data ?? [])
        .filter((recipe) => recipe.role === role && recipeFitsSystem(recipe, system.data))
        .sort((left, right) => (
          Number(right.certified) - Number(left.certified)
          || (left.total_size_bytes ?? Number.MAX_SAFE_INTEGER)
            - (right.total_size_bytes ?? Number.MAX_SAFE_INTEGER)
        ))
  );
  const performAction = (role: SetupRoleReadiness) => {
    if (role.next_action === "verify_generation") {
      verifyRole.mutate(role.role);
      return;
    }
    if (role.next_action === "retry_install" && role.job_id) {
      retryInstall.mutate(role.job_id);
      return;
    }
    if (role.next_action === "activate_model" && role.install_id) {
      activateModel.mutate(role.install_id);
      return;
    }
    if (
      ["install_runtime", "retry_runtime"].includes(role.next_action ?? "")
      && ["llama.cpp", "vllm", "comfyui"].includes(role.engine ?? "")
    ) {
      installRuntime.mutate(role.engine as RuntimeStatus["engine"]);
      return;
    }
    if (role.next_action === "restart_worker") {
      startWorker.mutate(role);
      return;
    }
    if (["repair_workflow", "review_workflow"].includes(role.next_action ?? "")) {
      onOpenWorkflows();
      return;
    }
    onOpenModels(role.role);
  };
  const actionLabel = (role: SetupRoleReadiness): string => {
    if (role.next_action === "verify_generation") return "Run quick test";
    if (role.next_action === "retry_install") return "Retry install";
    if (role.next_action === "activate_model") return "Re-check model";
    if (role.next_action === "install_runtime") return "Install runtime";
    if (role.next_action === "retry_runtime") return "Retry runtime";
    if (role.next_action === "restart_worker") return "Restart worker";
    if (["repair_workflow", "review_workflow"].includes(role.next_action ?? "")) {
      return "Review workflows";
    }
    return `Choose ${role.role} model`;
  };
  const pendingRole = startWorker.variables?.role;
  // One role at a time, one attempt per role: a load failure surfaces in the
  // error callout rather than looping, and the user can still prepare by hand.
  const autoPrepared = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!autoPrepare || startWorker.isPending) return;
    const preparable = report.roles.find(
      (role) =>
        role.checks.some((check) => check.action === "prepare_worker")
        && !autoPrepared.current.has(role.role),
    );
    if (!preparable) return;
    autoPrepared.current.add(preparable.role);
    startWorker.mutate(preparable);
  }, [autoPrepare, report, startWorker]);
  const error = installRecipe.error || retryInstall.error || installRuntime.error || startWorker.error || verifyRole.error || activateModel.error;

  return (
    <AccessibleDialog
      title={report.state === "ready" ? "Setup complete" : "Set up LM Atelier"}
      eyebrow="Local models"
      closeLabel="Close setup"
      onClose={onClose}
      className="setup-wizard"
    >
      <p className="setup-intro">
        Install each model with one click. Ready means activation passed and a quick local generation completed.
      </p>
      <div className="setup-role-grid" aria-live="polite">
        {report.roles.map((role) => {
          // A ready role that has not loaded its model still has something to
          // say, so the preparation note outranks the last passing check.
          const preparation = role.checks.find((check) => check.action === "prepare_worker");
          const issue = role.checks.find((check) => check.status !== "pass")
            ?? preparation
            ?? role.checks.at(-1);
          const job = jobs.data?.find((candidate) => candidate.id === role.job_id);
          const recipe = role.next_action === "select_model"
            ? suitableRecipes(role.role)[0]
            : undefined;
          const actionPending = (
            (retryInstall.isPending && retryInstall.variables === role.job_id)
            || (activateModel.isPending && activateModel.variables === role.install_id)
            || (installRuntime.isPending && installRuntime.variables === role.engine)
            || (startWorker.isPending && pendingRole === role.role)
            || (verifyRole.isPending && verifyRole.variables === role.role)
          );
          return (
            <article className={`setup-role ${role.state}`} key={role.role}>
              <header>
                <span className="setup-role-icon">
                  {role.role === "chat" ? <Bot /> : role.role === "image" ? <ImageIcon /> : <Film />}
                </span>
                <span><strong>{setupRoleName(role.role)}</strong><small>{role.state === "ready" ? "Ready" : role.state === "in_progress" ? "In progress" : "Action needed"}</small></span>
                {role.state === "ready" ? <Check aria-hidden="true" /> : role.state === "in_progress" ? <LoaderCircle className="spin" aria-hidden="true" /> : <Activity aria-hidden="true" />}
              </header>
              <p>{issue?.message ?? "Setup status is unavailable."}</p>
              {job && (
                <div className="setup-job">
                  <small>{jobProgressText(job)}</small>
                  <div
                    className="progress-track"
                    role="progressbar"
                    aria-label={`${setupRoleName(role.role)} setup progress`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={jobProgressFraction(job) === null
                      ? undefined
                      : Math.round(jobProgressFraction(job)! * 100)}
                  >
                    <div
                      className={jobProgressFraction(job) === null ? "indeterminate" : undefined}
                      style={jobProgressFraction(job) === null
                        ? undefined
                        : { width: `${jobProgressFraction(job)! * 100}%` }}
                    />
                  </div>
                </div>
              )}
              {role.state === "action_required" && (
                <div className="setup-actions">
                  {recipe && (
                    <button
                      className="primary compact-button"
                      disabled={installRecipe.isPending}
                      onClick={() => installRecipe.mutate(recipe.id)}
                    >
                      {installRecipe.isPending && installRecipe.variables === recipe.id
                        ? "Starting…"
                        : `Install ${recipe.name}`}
                    </button>
                  )}
                  {role.next_action && (
                    <button
                      className={recipe ? "secondary compact-button" : "primary compact-button"}
                      disabled={actionPending}
                      onClick={() => performAction(role)}
                    >
                      {actionPending ? "Working…" : actionLabel(role)}
                    </button>
                  )}
                  {recipe && <small>Reference candidate · {formatBytes(recipe.total_size_bytes)}</small>}
                </div>
              )}
              {preparation && role.state !== "action_required" && (
                <div className="setup-actions">
                  <button
                    className="secondary compact-button"
                    disabled={actionPending}
                    onClick={() => startWorker.mutate(role)}
                  >
                    {actionPending ? "Preparing…" : "Prepare now"}
                  </button>
                  <small>Or skip: the first request loads it instead.</small>
                </div>
              )}
            </article>
          );
        })}
      </div>
      {error && <ErrorCallout message={error.message} />}
      {recipes.error && <ErrorCallout message="Reference choices are temporarily unavailable. You can still browse the model library." />}
      <footer>
        <button className={report.state === "ready" ? "primary" : "secondary"} onClick={onClose}>
          {report.state === "ready" ? "Done" : "Not now"}
        </button>
      </footer>
    </AccessibleDialog>
  );
}

/** The installer's hand-off surface: setup before the workspace exists.
 *
 * Reached only via the installer's --first-run-setup launch. The workspace
 * is deliberately not rendered behind the dialog - the application becomes
 * available when setup finishes or the person explicitly skips, and both
 * exits run through onExit so the query flag is cleared exactly once.
 */
export function FirstRunSetup({
  report,
  onExit,
  onOpenModels,
  onOpenWorkflows,
}: {
  report: SetupReadinessReport;
  onExit: () => void;
  onOpenModels: (role: SetupRoleReadiness["role"]) => void;
  onOpenWorkflows: () => void;
}) {
  return (
    <div className="first-run-shell">
      <SetupWizard
        report={report}
        autoPrepare
        onClose={onExit}
        onOpenModels={onOpenModels}
        onOpenWorkflows={onOpenWorkflows}
      />
    </div>
  );
}
