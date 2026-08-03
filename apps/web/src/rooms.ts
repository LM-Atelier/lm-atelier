/** The workspace surfaces the sidebar can navigate to. */
export type View = "chat" | "media" | "models" | "workflows" | "studio" | "settings";

/** Which views are places you read and write in rather than make in.
 *
 * An atelier is defined by its light. Prose, settings, and a catalogue are
 * read, so they get a warm paper ground. The studio, the media grid, and
 * the workflow graph are where work is judged, and colour cannot be judged
 * against a warm ground - those stay in the dark room.
 *
 * The sidebar deliberately does not change. It is the building; only the
 * room does, so moving between them reads as moving rather than as the
 * page repainting itself.
 */
export const READING_ROOM_VIEWS: ReadonlySet<View> = new Set<View>([
  "chat",
  "models",
  "settings",
]);
