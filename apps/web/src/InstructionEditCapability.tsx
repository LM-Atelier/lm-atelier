export function InstructionEditCapability({ manifest }: { manifest: Record<string, unknown> }) {
  if (manifest.instruction_edit_capability !== "declared") {
    return <span className="muted">Instruction editing: unknown</span>;
  }
  return (
    <span className="muted">
      <span>Instruction editing: declared by source</span>{" "}
      <span>Requires a compatible workflow.</span>
    </span>
  );
}
