# Workflow recipes

A workflow recipe saves request settings for one use case. Create separate recipes
for chat, image generation, whole-image editing, masked inpainting, outpainting,
upscaling, video generation, or image animation. The request's operation and inputs
determine which recipe applies.

## Create or change a recipe

Open **Workflows**, choose **Manage recipes**, then **New recipe**. Give it a name
and choose its use case. Select a **Reference workflow for settings**, check the
settings to include, and adjust their values. Choose **Save recipe** when ready.

The reference supplies the available controls. Your workflow selections determine
which workflow runs, and compatibility is checked against its selected revision
when you send a request. Saving a recipe does not copy or modify a workflow graph.

Only included settings are saved. Existing saved settings that the reference
cannot expose remain listed until you remove them explicitly. Select a compatible
reference or remove an unsupported setting before using the recipe. If the setting
controls cannot load, use **Retry setting controls**; the draft is preserved.

## Choose where a recipe applies

Open **Recipes for this chat** below the chat's workflow selectors to choose a
recipe for each use case. A chat choice takes priority over its project's choice,
which takes priority over the workspace default.

- **Inherit project / workspace** uses the project's choice, then the workspace
  default when the project also inherits.
- **Automatic (no recipe)** skips recipes at that scope, including inherited
  recipes. The workflow selectors and explicit request settings still apply.
- A named recipe applies its included settings to that use case.

In **Manage recipes**, expand **Workspace defaults** to set a default for each use
case. Expand **Project choices** and choose a project to set its overrides. Choose
**Inherit workspace** there to restore inheritance.

Explicit settings on an individual request take priority over recipe settings.
Choosing a recipe does not change generation mode or the selections for other
workflow types. A recipe also cannot replace an explicitly selected workflow
family with a different family.

## Compatibility and earlier requests

The selected workflow must support the request's operation, required inputs and
every included setting. A recipe cannot grant mask, outpainting or upscaling
support to a workflow that lacks it. It also cannot make an untrusted or unready
workflow runnable. Unsupported settings cause a refusal instead of being silently
discarded. Choose a compatible workflow, change the recipe, or select
**Automatic (no recipe)** to continue without a recipe.

Chat recipes need a saved workflow revision with a bound chat profile. A legacy
chat profile that has no workflow revision can still be used with
**Automatic (no recipe)**.

Accepted requests retain a snapshot of the recipe settings they used. Editing or
retrying an earlier request keeps that snapshot instead of acquiring a newer
workspace default. A request accepted without a recipe continues without one.
Changing the operation or inputs can require a different use case and a new
request.

## Disable or delete a recipe

Remove the recipe's chat and project selections and its workspace default before
disabling or deleting it. Select **Edit** to change its enabled state, or **Delete**
and then **Confirm deletion** to remove it. Built-in recipes cannot be edited or
deleted. Earlier accepted requests retain their recorded settings.
