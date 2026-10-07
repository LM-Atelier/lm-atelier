import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";

// Each test starts with this browser's storage empty, as a fresh visit would,
// so nothing one test writes down is read back by the next.
afterEach(() => {
  localStorage.clear();
});
