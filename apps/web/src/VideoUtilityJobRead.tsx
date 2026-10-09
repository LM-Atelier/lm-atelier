import type { UseQueryResult } from "@tanstack/react-query";
import type { Job } from "./types";

/**
 * Says that an accepted video utility job could not be read, and reads it again when asked.
 *
 * The job keeps running whatever the page could read, so this never sends the work again:
 * the dialog keeps following the same job, and the button only asks how it is going.
 */
export function JobReadProblem({ job, subject }: { job: UseQueryResult<Job>; subject: string }) {
  if (!job.error) return null;
  const readAgain = () => {
    if (!job.isFetching) void job.refetch();
  };
  return (
    <div className="row-actions">
      <p role="alert">{subject} was accepted, but how it is going could not be read: {job.error.message}</p>
      <button type="button" className="secondary" aria-disabled={job.isFetching} onClick={readAgain}>
        Check again
      </button>
    </div>
  );
}
