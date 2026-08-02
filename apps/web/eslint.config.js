import js from "@eslint/js";
import globals from "globals";
import jsxA11y from "eslint-plugin-jsx-a11y-x";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist"] },
  {
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
      jsxA11y.configs.recommended,
    ],
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "jsx-a11y-x/label-has-associated-control": ["error", { assert: "either", depth: 3 }],
      "max-len": ["error", { code: 2607 }],
      "max-lines": ["error", { max: 4051 }],
      "max-lines-per-function": ["error", { max: 414 }],
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
    },
  },
  {
    // Test ceilings exist to stop unbounded drift, not to discourage coverage:
    // a cap that blocks adding a test is a cap pointed the wrong way. Raise
    // these when a change adds real cases; ratchet the source limits down.
    files: ["**/*.test.{ts,tsx}"],
    rules: {
      "max-len": ["error", { code: 362 }],
      "max-lines": ["error", { max: 5915 }],
      "max-lines-per-function": ["error", { max: 5620 }],
    },
  },
);
