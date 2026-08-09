/** The workspace surfaces the sidebar can navigate to.
 *
 * This used to also decide which room a view rendered in - prose on paper,
 * the studio in the dark. That read as one interface disagreeing with
 * itself, so the room is now the person's choice and applies to everything.
 * What is left is the list of places you can go.
 */
export type View = "chat" | "media" | "models" | "references" | "workflows" | "studio" | "settings";
