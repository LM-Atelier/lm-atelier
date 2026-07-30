import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Keeps a render failure from blanking the whole workspace.
 *
 * Everything below this renders inside one component tree, so an exception
 * thrown while rendering any part of it unmounts all of it. Without a boundary
 * that leaves an empty page with nothing to read and nothing to click, and the
 * only recovery is knowing to reload by hand.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // The console is the only place this can go; the app cannot be trusted to
    // render a report of its own failure.
    console.error("LM Atelier could not render", error, info.componentStack);
  }

  private reload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="render-error" role="alert">
        <h1>LM Atelier could not display this page</h1>
        <p>
          Your chats, models and generated media are stored locally and are not
          affected. Reloading usually clears this.
        </p>
        <button className="primary" onClick={this.reload}>
          Reload
        </button>
        <details>
          <summary>Technical details</summary>
          <pre>{error.message}</pre>
        </details>
      </div>
    );
  }
}
