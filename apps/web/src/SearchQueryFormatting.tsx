import "./ChatSearchConsent.css";

// Formatting affects spelling and emoji, so disclose it without rewriting the query.
const formatting = /[\p{C}\p{Z}\p{Default_Ignorable_Code_Point}\u2800]/u;
const names: Record<number, string> = {
  0x00ad: "soft hyphen",
  0x034f: "combining joiner",
  0x180e: "vowel separator",
  0x200b: "zero-width space",
  0x200c: "non-joiner",
  0x200d: "joiner",
  0xfe0e: "text style",
  0xfe0f: "emoji style",
  0x2800: "blank",
};

export function SearchQueryFormatting({ query, descriptionId }: { query: string; descriptionId: string }) {
  const marks: { code: number; position: number }[] = [];
  let position = 0;
  for (const character of query) {
    position += 1;
    if (position > 2000) { marks.length = 0; break; }
    if (character !== " " && formatting.test(character)) {
      marks.push({ code: character.codePointAt(0)!, position });
    }
  }
  return (
    <section className="chat-search-formatting" aria-label={marks.length ? "Query formatting" : undefined}>
      {marks.length > 0 && <>
        <p className="chat-search-query-text" aria-label="Query as entered" dir="auto">{query}</p>
        <p>Formatting marks are listed by character position. The original query is sent unchanged.</p>
      </>}
      <ul id={descriptionId} className="chat-search-formatting-preview" dir="ltr"
        aria-live="polite" aria-atomic="true">
        {marks.map(({ code, position }) => (
          <li className="chat-search-character" key={position}>
            Character {position}: {names[code] ?? "character"}{" "}
            {"U+" + code.toString(16).toUpperCase().padStart(4, "0")}
          </li>
        ))}
      </ul>
    </section>
  );
}
