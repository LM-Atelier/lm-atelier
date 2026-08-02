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
that flow workers are also loaded ahead of time, so the first request pays
nothing. Everything below applies the same way when setup runs inside the
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

Nothing about your chats or media leaves the machine. Model downloads reach the
model host, and that is the only network traffic setup needs.

## If a step will not complete

Start with [Troubleshooting](TROUBLESHOOTING.md), which lists the setup states
by name and what each one means. The most common first-run cases are a runtime
that is unavailable on the platform, not enough disk space for the selected
model, and a model that downloads but fails activation because it does not fit
in available memory.
