# Workflow packages

A ComfyUI workflow shared as a package can name custom nodes and Python
dependencies that this machine has never seen. LM Atelier will fetch and stage
all of it for you. What a package needs is shown before anything downloads.
Installing and enabling a verified ComfyUI Registry release can record trust
and activate it automatically. Code that needs review stays inactive until
you approve the exact package in **Prepared packages**.

## Reviewing before anything lands

Importing a workflow bundle opens a review of what the package needs: the
nodes it uses, the model files it references, the custom node packages it
depends on, and any findings that block an import outright - links that
connect to nothing, references that reach outside the model folders, formats
the app never loads. Opening the review changes nothing; installation starts
only when you choose **Install and enable** for a pinned package or approve
the model files it needs.

## Settings from the graph

When a workflow is created or imported, LM Atelier maps supported graph controls
to generation settings using the media runtime's node definitions. Saved values
become defaults, and controls retain the node's supported ranges and choices.
The application also applies its own limits; a saved value outside those limits
must be changed explicitly rather than silently replaced during import.

This includes size controls, sampling controls and supported video controls. A
workflow that uses an aspect-ratio choice, a megapixel budget and a rounding
multiple keeps those controls instead of gaining unrelated width and height
fields. Separate stages keep separate settings. Existing explicit input bindings
and the dedicated LoRA controls retain their own behavior.

Connections inside the graph are preserved. A value supplied by another node,
fixed by a primitive node or disconnected from the output is shown with an
explanation instead of an ineffective control. A dynamic choice that changes a
node's inputs must be changed in the native editor.

The media worker must be available when mapping a new workflow or changed graph.
This applies to both visual workflows and API-only graphs. An unchanged workflow
with saved mappings can be exported, imported or cloned while the worker is
offline. Saving a changed graph derives its mappings again, so controls do not
keep references to a previous revision.

## Preparing an exact version

When a needed package is not installed and pins exactly one version, the
review offers **Install and enable**. Installation is a normal job: it resolves the
package against the ComfyUI Registry, downloads the archive and every wheel
its dependencies need, verifies hashes end to end, and assembles an offline
environment. Progress shows in the jobs panel stage by stage, and the job can
be cancelled at any point.

Preparation first commits the downloaded package inactive and untrusted.
The installation then verifies its exact Registry identity, archive, files,
and dependency environment before recording trust and enabling eligible
releases. Unknown warnings, unreviewed commit sources, and an explicit trust
revocation require review; installation never silently overrides them.
Preparation needs the media worker stopped, so every preparation waits for
current media work, stops a running worker, and restarts it with its previous
setup when preparation ends.

## Trusting and activating

Prepared packages appear in the **Prepared packages** panel on the Workflows
page, each showing its identity hashes, current state, and any remaining decisions:

- **Trust** is a statement about you, not the package: that you reviewed this
  exact code and accept it running inside ComfyUI. Granting it re-verifies the
  package's files and dependencies first, and asks for explicit confirmation.
  Revoking trust needs no ceremony and also deactivates.
- **Activate** loads a trusted package into the media runtime. Activation
  re-verifies everything, then restarts the media worker with the package in
  place; if startup fails, the package is deactivated again and the prior
  runtime is restored. Deactivating restarts the runtime without it.

Both actions need the media worker stopped first, and both refuse with a
plain-language reason when something is not right - a package that is no
longer verifiable, a worker that is still running, activation of something
untrusted.

Pinned custom nodes installed by hand through the **Custom nodes** panel keep
their own separate review: an exact commit you trust explicitly, updated and
rolled back per revision.

## Where the pieces live

Prepared node packages and their wheel environments live inside the managed
data folder, isolated per package version and content-addressed by hash.
Deleting the app's data folder removes all of it; nothing is installed into a
system Python or a global ComfyUI.
