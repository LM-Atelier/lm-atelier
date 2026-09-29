# Editing studio

The editing studio turns common image edits into one-click choices: pick a
template, and its complete instruction lands in the composer, visible and
editable before anything runs. The send stays an ordinary edit turn - it can
be reviewed, revised, retried, and compared like any other.

## Reaching the studio

- **From the composer** - attach an image and choose *Open editing studio*.
- **From a chat** - hover any image and choose *Edit*; the image attaches and
  the studio opens over it.
- **From the Media library** - every image card has an *Edit* action, and a
  multi-select: tick several images and *Edit together in the studio* hands
  the whole selection to the composer at once. When no chat exists yet, one
  is created first.

## Templates

The built-in set covers everyday edits - style transformations like
watercolor, oil, pencil, and anime; colorizing; restoring old photos;
relighting; removing text - and each template accepts an optional line of
detail ("focus on the harbor in the background") that slots into its
instruction. Templates you find yourself repeating can be kept: with a draft
instruction in the composer, the studio offers **Save as template** under the
name you give it.

## Editing many images at once

With several images attached, the studio's **Apply to each** sends one
independent edit turn per image. Each runs, verifies, and can be retried on
its own; one failure never abandons the rest.

## While an edit runs

An edit that runs a model shows, under **Apply**, the step it has reached and a
bar that fills as the work goes on: the same progress a chat shows for its own
pictures.

## Seeing what changed

- **Compare** slides the edited result over its source, so differences show
  where they are.
- **Lineage** appears once a result is at least two edits deep and walks the
  whole chain, oldest first: the image that entered each step and the exact
  instruction that transformed it, ending at the current result. A chat
  forked from another carries results, not history, and the lineage view
  never pretends otherwise.

## Exact edits in Image Studio

Some changes need no model at all, and Image Studio makes them itself, from
the picture's stored bytes. They run without a graphics card or an installed
workflow, each arrives as the next step like any other result, and its record
names what was done rather than a model:

- **Rotate, straighten or flip** turns or mirrors the whole picture. Straighten
  turns it a few degrees either way, shown on the picture as the slider moves,
  and keeps the largest box of the picture's own shape that the turned picture
  still covers, so no empty corners appear.
- **Crop** keeps a box you draw, which can be held to a shape as you draw it:
  square, 4:3, 3:2 or 16:9 either way up, or the picture's own shape. A box
  already drawn takes a newly chosen shape at once. **Resize** changes the size
  in pixels, in proportion unless you switch that off; **Change the canvas
  size** adds room around the picture or trims it, from one of nine places,
  filled with transparency, white or black.
- **Adjust light and color** has brightness, contrast, saturation, warmth and
  tint sliders, shown on the picture as they move. Warmth balances blue against
  amber, and tint green against magenta. The preview is the result: the browser
  and the app work the adjustment out the same way.
- **Blur**, **Pixelate** and **Paint** act on a marked area, brushed with the
  tool itself or selected first with any selection tool. Everything outside the
  marking is left exactly as it was. Pixelate breaks the area into square
  blocks of the size you choose, each the average of the pixels it covers.
  Paint shows its color and opacity on the marking while you choose them.
- **Add text** draws your words in one of the app's typefaces, at a size, color
  and place you choose, with an optional outline or shadow so they read over a
  busy picture. What the canvas shows is what is added.

**Export** saves the picture as stored, or as PNG, JPEG or WebP. Exports come
out upright and keep the picture's color profile; JPEG has no transparency, so
transparent parts come out white.
