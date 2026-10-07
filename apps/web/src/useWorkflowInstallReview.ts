import { useState } from "react";
import type { WorkflowInstallOffer } from "./types";

export function useWorkflowInstallReview() {
  const [installReview, setInstallReview] = useState<{ offer: WorkflowInstallOffer; name: string } | null>(null);
  const [downloadsQueued, setDownloadsQueued] = useState(false);
  const reviewInstall = (offer: WorkflowInstallOffer, name: string) => {
    setDownloadsQueued(false);
    setInstallReview({ offer, name });
  };
  return { installReview, setInstallReview, downloadsQueued, setDownloadsQueued, reviewInstall };
}
