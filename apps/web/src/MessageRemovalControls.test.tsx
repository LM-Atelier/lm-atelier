import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import { UserMessageControls } from "./MessageRemovalControls";

it("confirms that one item is removed while replies stay", () => {
  const remove = vi.fn();
  const deleteExchange = vi.fn();
  render(
    <UserMessageControls
      messageId="msg_constructed"
      createdAt="2026-08-25T12:00:00Z"
      copyableText="Constructed message"
      onRemoveItem={remove}
      onDeleteExchange={deleteExchange}
    />,
  );

  fireEvent.click(screen.getByRole("button", { name: "Remove this item, keep replies" }));
  expect(remove).not.toHaveBeenCalled();
  expect(screen.getByText("Only this item's content is removed. Replies stay.")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Remove this item, keep replies" }));

  expect(remove).toHaveBeenCalledWith("msg_constructed");
  expect(deleteExchange).not.toHaveBeenCalled();
});
