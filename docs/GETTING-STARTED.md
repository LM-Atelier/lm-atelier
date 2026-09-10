# Getting started

This walks through a first run: what LM Atelier needs, the order it sets things
up in, and how to tell that a role is genuinely working rather than merely
installed.

LM Atelier does not ship model weights or inference engines. On first launch it
has the application only, and setup fills in the rest.

## Before you start

Check what your machine can run. A role only works if its engine is available
for your platform:

| Role | Engine | Available on |
| --- | --- | --- |
| Chat and vision | llama.cpp | Windows 11 x64, Ubuntu 24.04 LTS x64 |
| Image and video | ComfyUI | Windows 11 x64 with a compatible NVIDIA GPU |

Image and video have no automatic setup on Linux, or on Windows without a
compatible NVIDIA GPU. On those machines you can use chat and vision, and you
can point LM Atelier at an externally configured media engine, which is not
covered here and is not certified.

You also need free disk space. The ComfyUI runtime needs roughly 2 GB to
download and 8 GB free to install. Models range from a few gigabytes to more
than twenty. The setup panel only offers models that fit your reported memory
and disk.

## The order setup happens in

Setup works in one direction, and each step depends on the one before it.

1. **Runtime.** The engine for the role is downloaded and verified against a
   pinned checksum. This happens automatically the first time a role needs it,
   and you can start it yourself from the setup panel.
2. **Model.** You choose a model. Before anything transfers you are shown what
   it will cost - the download size, your free space, and the memory it needs to
   load - and the files are then checked against expected hashes as they
   arrive.
3. **Activation.** The model is loaded and asked to produce one tiny output.
   Only after that does LM Atelier consider it usable. A model that downloads
   but fails this step is not ready, and the panel says so. If activation later
   goes stale - an update can change the runtime or workflow contract -
   **Re-check model** proves it again without re-downloading.
4. **Quick test.** A single small generation runs through the normal queue, end
   to end. This is what turns a role green.

The setup panel checks these in order and shows you the first thing that is not
satisfied, with an action when there is one. If the runtime for a role is
unavailable on your machine, it tells you at the start rather than after a large
download.

## First run

Installing from the packaged installer flows straight into setup: the
workspace appears only after the roles you chose are ready or you skip. In
that flow workers can also be loaded ahead of time to reduce the initial wait.
Generation time still depends on the model, hardware, and request. Everything below applies the same way when setup runs inside the
application.

1. Launch LM Atelier. The setup panel opens by itself when anything is
   incomplete. If you dismiss it, reopen it from **Setup** in the sidebar.
2. Work one role at a time, starting with chat. Each card shows the current
   state and a single action.
3. When a card offers a recommended model, that choice already fits your
   reported memory and disk. You can instead browse the full catalog from the
   model library.
4. Wait for the download. Progress is shown on the card, and when several
   downloads run at once one line above the cards totals what is left and,
   while transfer rates are fresh, roughly how long it will take. Downloads
   survive a restart.
5. Run the quick test when the card offers it. A role is only **Ready** once a
   real local generation has completed.

Repeat for image and video if your machine supports them.

A ready role whose model is not loaded yet says so - the first request would
wait while it loads - and offers **Prepare now** to pay that wait immediately
instead. Skipping is fine; nothing breaks either way.

## Knowing a role really works

"Installed" and "working" are different states, and LM Atelier distinguishes
them deliberately:

- **Not runtime verified** means the files are present but the model has not
  produced output on this machine. It is not usable yet.
- **Ready** means a bounded generation completed with this exact model, profile,
  and workflow on this hardware.

A role can return to an unverified state after an update, if its files change,
or if the runtime is replaced. That is expected, and it means the previous proof
no longer describes the current setup.

## Where things go

Models, runtimes, generated media, and the database live in the local data
folder listed in [Troubleshooting](TROUBLESHOOTING.md). Updates keep it.

Chats and media are stored locally by default. Catalog requests, model downloads,
and runtime downloads contact their providers. Configured adapters and custom
nodes can make their own network requests; review their behavior before using
them. See [Privacy and local data](PRIVACY.md).

## Keeping installed models current

The Model library's **Check for updates** asks the provider - only when you
press it, never in the background - whether any installed asset with an exact
recorded version has something newer. The report keeps its three answers
separate: updates available, up to date, and could not check.

Supported LoRA and checkpoint updates offer **Review update**. Checkpoint review
uses the installed model's chat, image, or video role; if that role cannot be
determined, the app asks you to check for updates again. Other asset types direct
you to the catalog. Downloads use the normal installation checks; nothing
updates itself or silently switches a profile to a newer model.

## If a step will not complete

Start with [Troubleshooting](TROUBLESHOOTING.md), which lists the setup states
by name and what each one means. The most common first-run cases are a runtime
that is unavailable on the platform, not enough disk space for the selected
model, and a model that downloads but fails activation because it does not fit
in available memory.

## Explore the workspace

LM Atelier brings conversations, models, workflows, and generated media into one
local workspace. Configure a compatible model with [Getting started](GETTING-STARTED.md)
before trying real generation. Features described here follow the current source;
an older installer may expose fewer controls.

### Start a conversation

Create a chat and choose a configured model or **Auto**. Auto selects among the
resources available to the application; installing a model and verifying that it
can run are separate steps. A request that needs an unavailable resource reports
what must be configured.

Try a simple text request first. For example: "Suggest three names for a mountain
photography collection." Then try an image request with a ready image setup:
"Create a watercolor landscape with a lake and distant mountains."

Images and video appear in the conversation, with queue progress while generation
runs. Output count and seed controls let you vary a request. Generation speed and
supported sizes depend on the selected model, workflow, and hardware.

### Watch accepted work

Choose **View accepted work** in the Jobs panel to see active generation, transfers,
and installs. Plans keep their jobs grouped. Use **Work category** to narrow the
list and **Load more accepted work** to browse additional items. The count shows
how many active items you have loaded out of the total.

Items are ordered by acceptance time. Different resources can run at the same
time. Plans show completed-step counts and running, queued, and paused job
counts. Choose **Show steps** to inspect a plan's individual steps and unfinished
prerequisites; **Load more steps** continues a long plan.

The view updates automatically and shows when it was last checked. Use
**Refresh** for a fresh list. If an update fails, an error warns when
previously loaded items are still shown. Step details have their own
**Retry step details** action.

### Browse workflows

Open **Workflows** to browse families and their variants. Search names, descriptions,
tags, use cases, or recorded dependency names; filter by operation, readiness,
or source, and sort by name or readiness. A family groups related choices so their purpose and requirements can
be compared together.

Select a variant to inspect its revision and dependency details. Readiness labels
explain whether it can run, needs setup, needs review, or is unavailable. Setting
a default does not install missing dependencies or grant trust. The chat workflow
selector also lists choices that need attention and explains why they cannot run.

When a variant offers **Review downloads**, inspect its required files and total
download size, then choose **Download reviewed files** to start. The app checks
the current files and plans again before downloading. A dependency count is a
summary of recorded requirements; use the readiness details to see what still
needs attention.

Open **Show revision history** to inspect saved versions and their dates. When
viewing an older version, **Show changes** compares its saved graph, controls,
and dependencies with the current revision. Large comparisons report when their
display limit is reached. **Restore as new revision** is a separate action.
Select the current revision to use **Validate**; its result applies to that
revision.

Names, tags, and use-case descriptions can be edited. Automatically derived
use-case text is labeled, and your explicit text edits take precedence.
Archiving hides a family from ordinary browsing; include archived families when
you want to find or restore one.

See [Workflow packages](WORKFLOW-PACKAGES.md) before importing code or activating
custom nodes.

### Reuse a prompt

**Prompt Templates** let you save a reusable request with selectable values and
resource choices. A template can also ask the configured local chat model to fill
slots. Model-filled templates require a working chat setup as well as the setup
needed for the requested media.

Review the template settings and output count before submitting. Expansion errors
are shown in the dialog; they do not mean media was generated. Once a request is
queued, its outputs stay grouped in the conversation. Creating or editing a
template by itself does not start a generation.

### Work with media

Use the media library to find generated outputs and manage reusable image
references. Select an image when composing a request that uses a reference.
Supported image edits open in the [editing studio](EDITING-STUDIO.md), where the
before/after view helps compare the result.

Reference support depends on the workflow. Merely selecting an image does not
make every model or workflow capable of using it.

### Prepare a demonstration

1. Use a fresh chat and neutral sample requests that you are comfortable sharing.
2. Finish setup and run the quick test for each role you intend to demonstrate.
3. Prepare the required models before presenting; downloads and model loading
   can take longer than the generation itself.
4. Show a short conversation, one supported generation, its library entry, and
   the workflow's readiness details.
5. Check visible names, prompts, images, and paths before sharing your screen.

For a source-only interface demonstration without model downloads, the mock
engines in the [development setup](../README.md#local-development) produce sample
responses. They do not demonstrate model quality or hardware performance.

See [Troubleshooting](TROUBLESHOOTING.md) for setup and queue failures, and
[Privacy and local data](PRIVACY.md) for network, backup, and deletion behavior.
