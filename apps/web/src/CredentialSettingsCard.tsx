import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { CredentialProvider } from "./types";

type CredentialSettingsCardProps = {
  provider: CredentialProvider;
  providerLabel: string;
  description: string;
  environmentVariable: string;
  placeholder: string;
};

export function CredentialSettingsCard({
  provider,
  providerLabel,
  description,
  environmentVariable,
  placeholder,
}: CredentialSettingsCardProps) {
  const client = useQueryClient();
  const [token, setToken] = useState("");
  const credential = useQuery({
    queryKey: ["credential", provider],
    queryFn: () => api.credentialStatus(provider),
  });
  const saveCredential = useMutation({
    mutationFn: () => api.setCredentialToken(provider, token),
    onSuccess: (value) => {
      setToken("");
      client.setQueryData(["credential", provider], value);
    },
  });
  const removeCredential = useMutation({
    mutationFn: () => api.deleteCredentialToken(provider),
    onSuccess: (value) => client.setQueryData(["credential", provider], value),
  });
  const status = credential.data?.configured
    ? `Configured - ${credential.data.source.replace("credential_vault", "credential vault")}`
    : "Not configured";

  return (
    <section data-testid={`credential-${provider}`}>
      <div className="detail-title">
        <div><h2>{providerLabel} access</h2><p>{description}</p></div>
        <span className={`badge ${credential.data?.configured ? "tested" : ""}`}>{status}</span>
      </div>
      <div className="preset-create">
        <input
          aria-label={`${providerLabel} access token`}
          type="password"
          autoComplete="off"
          placeholder={placeholder}
          value={token}
          onChange={(event) => setToken(event.target.value)}
          disabled={credential.data?.source === "environment"}
        />
        <button
          className="primary"
          aria-label={`Save ${providerLabel} token`}
          disabled={!token.trim() || saveCredential.isPending || credential.data?.source === "environment" || credential.data?.vault_available === false}
          onClick={() => saveCredential.mutate()}
        >
          {saveCredential.isPending ? "Saving..." : "Save token"}
        </button>
        {credential.data?.configured && (
          <button
            className="secondary danger"
            aria-label={`Remove ${providerLabel} token`}
            disabled={removeCredential.isPending || credential.data.source === "environment"}
            onClick={() => removeCredential.mutate()}
          >
            {removeCredential.isPending ? "Removing..." : "Remove"}
          </button>
        )}
      </div>
      {credential.data?.source === "environment" && (
        <p className="muted runtime-note">The {environmentVariable} environment variable currently takes precedence. Unset it before managing the token here.</p>
      )}
      {credential.data && !credential.data.vault_available && (
        <div className="callout error" role="alert">No supported operating-system credential vault is available. Configure one or use {environmentVariable} for this process.</div>
      )}
      {(credential.error || saveCredential.error || removeCredential.error) && (
        <div className="callout error" role="alert">{(credential.error || saveCredential.error || removeCredential.error)?.message}</div>
      )}
    </section>
  );
}
