import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import { ErrorBoundary } from "./ErrorBoundary";
import { WorkspaceLockGate } from "./WorkspaceLockGate";
import { isWorkspaceLockRefusal } from "./workspaceLockState";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      // A locked workspace refuses the retry exactly as it refused the first try.
      retry: (failureCount, error) => failureCount < 1 && !isWorkspaceLockRefusal(error),
      refetchOnWindowFocus: false,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <WorkspaceLockGate>
          <App />
        </WorkspaceLockGate>
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
);
