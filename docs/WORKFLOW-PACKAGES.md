# Workflow packages

A ComfyUI workflow shared as a package can name custom nodes and Python
dependencies that this machine has never seen. LM Atelier will fetch and stage
all of it for you, but it never lets any of it run on someone else's say-so:
what a package needs is shown before anything downloads, and downloaded code
stays inert until you explicitly trust it and separately activate it.

## Reviewing before anything lands

Importing a workflow bundle opens a review of what the package needs: the
nodes it uses, the model files it references, the custom node packages it
depends on, and any findings that block an import outright - links that
connect to nothing, references that reach outside the model folders, formats
the app never loads. Nothing is installed, executed, or trusted from this
dialog.

## Preparing an exact version

When a needed package is not installed and pins exactly one version, the
review offers **Prepare**. Preparation is a normal job: it resolves the
package against the ComfyUI Registry, downloads the archive and every wheel
its dependencies need, verifies hashes end to end, and assembles an offline
environment. Progress shows in the jobs panel stage by stage, and the job can
be cancelled at any point.

A successful preparation is always committed **inactive and untrusted**.
Preparation needs the media worker stopped; a running worker refuses the job
rather than racing it.

## Trusting and activating

Prepared packages appear in the **Prepared packages** panel on the Workflows
page, each showing its identity hashes and the two decisions it is waiting
for:

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
