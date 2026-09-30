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

Leaving the studio for another view, for instance to install a workflow a tool
needs, keeps what you were doing. Come back to the same picture and the tool,
its selection and settings, and the words you wrote are as you left them. If a
newer result is on the canvas by then, the words come back and the selection
does not, since it was drawn on the earlier picture. This lasts while the app
stays open.

## Templates

The built-in set covers everyday edits - style transformations like
watercolor, oil, pencil, and anime; colorizing; restoring old photos;
relighting; removing text - and each template accepts an optional line of
detail ("focus on the harbor in the background") that slots into its
instruction. Templates you find yourself repeating can be kept: with a draft
instruction in the composer, the studio offers **Save as template** under the
name you give it.

In Image Studio, a result a model made can be kept the same way: with it on the
canvas, **Save this edit as a recipe** under the Recipes list records its words
and what its run actually used, the workflow, the model and the settings, but
never the selection or the seed. Exact edits made without a model offer no
recipe.

## Editing many images at once

With several images attached, the studio's **Apply to each** sends one
independent edit turn per image. Each runs, verifies, and can be retried on
its own; one failure never abandons the rest.

## While an edit runs

An edit that runs a model shows, under **Apply**, the step it has reached and a
bar that fills as the work goes on: the same progress a chat shows for its own
pictures. **Stop the edit** ends it, and the picture stays as it was.

## Seeing what changed

- **Compare** slides the edited result over its source, so differences show
  where they are.
- In Image Studio, **Hold to compare** shows the picture a result was made
  from while it is held, **Split** lays the two across a divider, and **What
  changed** tints every pixel the edit changed, from amber for a slight change
  to red for a large one, and says how much of the picture that is. What
  changed is offered when the result and the picture it was made from are
  exactly the same size, since only then do their pixels line up one to one.
  **Side by side** puts the two next to each other instead, whatever their
  shapes, so an extended or cropped result can be set beside what it was made
  from. Zooming or dragging either half moves both, and the same part of the
  scene stays in view in each.
- **Lineage** appears once a result is at least two edits deep and walks the
  whole chain, oldest first: the image that entered each step and the exact
  instruction that transformed it, ending at the current result. A chat
  forked from another carries results, not history, and the lineage view
  never pretends otherwise.

## Choosing a tool

Image Studio's tools sit in five runs down its left edge: exact edits made
without a model; selecting part of the picture; edits said in words; the
subject and the scene around it; and enlarging with detail restored. Select is
one tool. Once it is chosen, the panel offers the ways of drawing the
selection: the brush, the eraser, a rectangle, a lasso, a fill, or similar
colors. Choosing Select again returns to the way last used.

## Exact edits in Image Studio

Some changes need no model at all, and Image Studio makes them itself, from
the picture's stored bytes. They run without a graphics card or an installed
workflow, each arrives as the next step like any other result, and its record
names what was done rather than a model:

- **Rotate, straighten or flip** turns or mirrors the whole picture. Straighten
  turns it a few degrees either way, shown on the picture as the slider moves,
  and keeps the largest box of the picture's own shape that the turned picture
  still covers, so no empty corners appear.
- **Correct the perspective** squares up something photographed at an angle,
  such as a page or the front of a building. Drag the four corners on the
  picture onto its corners, and applying makes it the whole picture, upright.
  The panel names the size first: the longer of each pair of opposite sides.
- **Crop** keeps a box you draw, which can be held to a shape as you draw it:
  square, 4:3, 3:2 or 16:9 either way up, or the picture's own shape. A box
  already drawn takes a newly chosen shape at once. **Resize** changes the size
  in pixels, in proportion unless you switch that off; **Change the canvas
  size** adds room around the picture or trims it, from one of nine places,
  filled with transparency, white or black.
- **Adjust light and color** has brightness, contrast, highlights, shadows,
  saturation, vibrance, warmth, tint, sharpness, vignette and grain sliders,
  shown on the picture as they move. **Auto** sets warmth, tint, brightness and
  contrast from the picture itself. It takes a color cast out of the
  near-grey middle tones, brings the middle of the picture's brightness toward
  a middle grey, and widens a picture that uses only part of the range, each
  within a moderate reach, and leaves the other sliders as they were. A
  **look** (Vivid, Soft, Warm, Cool, Mono, Faded, Dramatic or Film) sets every slider
  to a named starting point. Either way the sliders show what was set, and
  any of them can still be moved before applying. Vibrance richens muted
  colors more than vivid ones, so a picture livens without its brightest
  colors clipping. Warmth balances
  blue against amber, and tint green against magenta. Sharpness crisps edges
  above zero and softens them below; where a picture is transparent, colors
  hidden there play no part. The vignette darkens the edges above zero and
  lightens them below, most at the corners, and leaves the middle as it was.
  Grain, from zero up, adds film-like grain: each pixel moves a little up or
  down, the same on all three colors, so the picture keeps its light and a grey
  stays grey. The preview is the result: the browser and the app work the
  adjustment out the same way.
- **Blur**, **Pixelate** and **Paint** act on a marked area, brushed with the
  tool itself or selected first with any selection tool. Everything outside the
  marking is left exactly as it was. Pixelate breaks the area into square
  blocks of the size you choose, each the average of the pixels it covers.
  Paint shows its color and opacity on the marking while you choose them.
- **Add text** draws your words in one of the app's typefaces, at a size, color
  and place you choose, with an optional outline or shadow so they read over a
  busy picture. Once written, drag them anywhere on the picture, or move them
  with the arrow keys after Enter takes hold, and turn them about their middle
  with the Turn slider. Choosing a place again puts them back there. What the
  canvas shows is what is added.

**Export** saves the picture as stored, or as PNG, JPEG or WebP. JPEG and WebP
come at a quality you choose, from best quality down to smallest file, trading
detail for size; PNG keeps every pixel. Exports come out upright and keep the
picture's color profile; JPEG has no transparency, so transparent parts come out
white.

**Use in chat** attaches the picture on screen to a chat's next message, as a
reference. It goes to the chat the studio was opened from, or else the chat
that was open, or else a new chat. Nothing is sent, and the chat's mode stays
as it was.

## Extending a picture

**Extend past the edge** paints beyond the picture's edges with an installed
outpainting workflow. Drag any edge of the frame outward, or pick an edge and
use the arrow keys; each edge can reach up to twice the picture's width or
height. The frame is drawn around the picture at whatever zoom you are
viewing, with the added area tinted, and the panel names the size the extended
picture will be.

The panel can also set the edges for you. **Extend to a shape** (Square, 4:5,
3:2, 16:9 or 9:16) adds canvas evenly on the two edges that grow, never cutting
the picture: a picture wider than the shape grows taller, and a taller one
wider. A shape the picture already is, or one that would need more than twice
the picture on an edge, cannot be chosen. **Extend to this size** takes an exact
width and height, no smaller than the picture, and puts the new room away from
where you place the picture. Either way the edges can still be dragged
afterwards.
