import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { Chat, WebSettings } from "./types";
import "./ChatWebAccess.css";

export function ChatWebAccess({ chat }: { chat: Chat }) {
  const client = useQueryClient();
  const configuration = useQuery({
    queryKey: ["web-search", "configuration"], queryFn: api.searchConfiguration,
  });
  const settings: WebSettings = {
    allow_url_fetch: false, allow_search: false, allow_search_without_asking: false,
    ...chat.web_settings_json,
  };
  const save = useMutation({
    mutationFn: (value: WebSettings) => api.updateChat(chat.id, { web_settings_json: value }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["chat"] });
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
  });
  const enabled = configuration.data?.installation_enabled === true;
  const searchReady = enabled && configuration.data?.configured === true;
  const change = (key: keyof WebSettings, checked: boolean) => {
    if (save.isPending) return;
    save.mutate({
      ...settings, [key]: checked,
      ...(key === "allow_search" && !checked ? { allow_search_without_asking: false } : {}),
    });
  };
  return (
    <details className="chat-web-access">
      <summary>Web access</summary>
      <fieldset aria-disabled={save.isPending} aria-busy={save.isPending}>
        <legend>Permissions for this chat</legend>
        <label><input type="checkbox" checked={settings.allow_url_fetch}
          disabled={!enabled && !settings.allow_url_fetch}
          onChange={(event) => change("allow_url_fetch", event.target.checked)} />
          Read links I include in messages</label>
        <label><input type="checkbox" checked={settings.allow_search}
          disabled={!searchReady && !settings.allow_search}
          onChange={(event) => change("allow_search", event.target.checked)} />
          Allow web searches</label>
        <label><input type="checkbox" checked={settings.allow_search_without_asking}
          disabled={(!searchReady || !settings.allow_search) && !settings.allow_search_without_asking}
          onChange={(event) => change("allow_search_without_asking", event.target.checked)} />
          Allow searches without asking again</label>
      </fieldset>
      <p>Each search shows its query and provider. Automatic searches wait five seconds so you can cancel.</p>
      {configuration.isPending && <p role="status">Checking web access…</p>}
      {configuration.data && !enabled && <p>Web access is turned off for this installation.</p>}
      {enabled && !searchReady && <p>Set up the search provider in Settings before enabling searches.</p>}
      {searchReady && <p>Search provider: CRW at {configuration.data?.provider_endpoint}</p>}
      {(configuration.error || save.error) && <p role="alert">{(configuration.error || save.error)?.message}</p>}
    </details>
  );
}
