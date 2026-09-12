import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { CredentialSettingsCard } from "./CredentialSettingsCard";

export function WebSearchSettings() {
  const configuration = useQuery({
    queryKey: ["web-search", "configuration"], queryFn: api.searchConfiguration,
  });
  return (
    <>
      <section>
        <h2>Web search</h2>
        <p>Connect your CRW search service, then enable web access separately in each chat.</p>
        {configuration.data && (
          <>
            <p>{configuration.data.installation_enabled
              ? "Web access is available for this installation."
              : "Web access is turned off for this installation."}</p>
            <p>{configuration.data.configured
              ? `Search provider: CRW at ${configuration.data.provider_endpoint}`
              : configuration.data.error_code === "search_provider_invalid"
                ? "The configured search provider address is invalid."
                : "No search provider is configured."}</p>
          </>
        )}
        <details><summary>Connection setup</summary>
          <p>Set LOCAL_LM_CRW_ENDPOINT to your CRW service address and restart LM Atelier.
            Use HTTPS, or HTTP with a loopback IP address for a service on this computer.</p>
          <p>LOCAL_LM_WEB_ACCESS_ENABLED=true makes web access available to chats.
            Keep it false to prevent all web access. Chat permissions cannot override it.</p>
          <p>If your service requires a token, save it below. Search results are never opened automatically.</p>
        </details>
        {configuration.isPending && <p role="status">Checking search configuration…</p>}
        {configuration.error && <p role="alert">{configuration.error.message}</p>}
      </section>
      <CredentialSettingsCard provider="crw" providerLabel="CRW"
        description="Store the token for your search service in the operating-system credential vault."
        environmentVariable="LOCAL_LM_CRW_TOKEN" placeholder="CRW API token" />
    </>
  );
}
