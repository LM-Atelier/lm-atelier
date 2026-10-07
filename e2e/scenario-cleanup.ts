export async function withScenarioCleanup<T>(scenario: () => Promise<T>, cleanup: () => Promise<void>): Promise<T> {
  let failed = false;
  let primary: unknown;
  try {
    return await scenario();
  } catch (error) {
    failed = true;
    primary = error;
    throw error;
  } finally {
    try {
      await cleanup();
    } catch (secondary) {
      if (!failed) throw secondary;
      const combined = new AggregateError([primary, secondary],
        primary instanceof Error ? primary.message : "The browser scenario failed.", { cause: secondary });
      if (primary instanceof Error) {
        combined.name = primary.name;
        combined.stack = primary.stack;
      }
      throw combined;
    }
  }
}
