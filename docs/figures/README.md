# Figure Sources

## Modeling pipeline

`wq_pipeline.dot` is the editable source for the modeling-pipeline figure
(`fig:modeling_workflow` in the manuscript). It supersedes
`wq_modelling_workflow.dot`, which was four full-width bands of heading plus detail
line: all of its content was linguistic and its edges carried no information.

### Resolution

The output is `wq_pipeline.png` at **1200 dpi** (7675 px wide, ~1.2 MB), copied into
`docs/report/draft/figures/` for the submission bundle.

This figure is the hardest case for a raster: line art plus 7.5 pt type, where every
cell border is a hairline. At low pixel density the rasteriser spreads each hairline
over a fraction of a pixel, which reads as graininess. The fix is pixel density, not
an antialiasing setting — 600 dpi was visibly grainy in the detail text, which sits at
the printed-size floor and has no margin to lose. MDPI asks for >=1000 dpi for line
art and combination figures, so 1200 dpi is both the fix and the requirement.

**Do not drop below 1000 dpi to save file size.** Cairo is the rasteriser
(`-Tpng:cairo`); the plain `-Tpng` backend is worse on hairlines at any density.

### What the diagram encodes

Five stages in the conventional data-science sequence, indexed by a navy rail down
the left edge. The rail replaces full-width header bars: those cost ~16 pt of height
each and made five unrelated stages look like five instances of one thing. Content
boxes are sized to their content, so the silhouette varies stage to stage.

- **The stage 3 pictogram** is the load-bearing element: columns are hours, the
  window ends *on* the sampling hour (hence no lead time), one predictor row is
  gapped because incomplete coverage is what costs laboratory samples downstream,
  and the target row shows the change between two laboratory measurements.
- **The train/test split** is a property of the data, not a stage of work, so it is a
  14 pt bar inside stage 3 rather than a band of its own.
- **The two lanes in stage 4** are the substance of that stage: the three
  machine-learning families go through subset search and rolling-origin
  cross-validation; the statistical reference models do not. That asymmetry is why
  reported accuracy carries selection optimism for one lane and not the other, and a
  stacked-band layout hides it.
- **The two lanes are styled in parallel**, because the seven method families are
  parallel: one box per family, same header font on both sides, with the reference
  models simply carrying shorter descriptions. "Statistical reference model" is the
  manuscript's own term for all four, so the figure uses it.
- **Within the machine-learning lane, process is styled apart from model.** The subset
  search and the cross-validation sit in tinted boxes under a "selection and
  validation" label; the three model families sit in white boxes under a "model
  families" label. With one fill and one header style throughout, "Predictor-subset
  search" read as a fourth model.
- The lane caption is "no beam search or CV tuning", not "untuned" — MLR *is* fitted,
  with its own embedded selection (Spearman, mutual information, Lasso, VIF) over
  three aggregation windows. What all four reference models share is that they skip
  the beam search and the rolling-origin tuning, and that is what the caption says.
- **The rolling-origin stair** shows ordered folds fitting and then scoring the next
  block.

Colour carries one meaning, and it tracks the **data**, not the methods. Navy = stage
index and the sampling hour. Blue = predictors and the train split. Amber = the
target and the test side. Neutral grey = absent data, and the untuned reference lane
in stage 4.

Amber additionally distinguishes **fill from border**, because it has to cover both a
kind of data and the stage that acts on it:

- **Amber fill** = a target-side *data object*: the Laboratory source in stage 1, the
  `Δ since previous` chip in stage 2, the prior-value and lab-sample rows and the
  `Δ target` bar in stage 3, the TEST segment of the split.
- **Amber border, white fill** = a *procedure* operating on the test side: the Basis
  and Metrics boxes in stage 5.

Without that split, the Laboratory card in stage 1 and the stage 5 boxes carried
identical fills while being entirely different kinds of thing. The reference lane in
stage 4 was also amber in an earlier draft, which made amber mean "reference method"
on top of everything else; it is grey now, so blue vs grey in stage 4 says exactly one
thing: tuned vs untuned. Both splits and both lanes are named in words as well as
coloured, so the figure survives greyscale printing and colour-vision deficiency.

Text is limited to headings, noun phrases and numbers. Result values never appear,
and neither do repository identifiers — the audience for this figure will never see
the code.

Three things were tried and removed. Per-source coverage strips in stage 1 read as a
loose qualitative sample rate rather than as coverage, and carried no decision. The
four-level support verdict classifies results, not data, and belongs with the results
table. A 2x2 matrix of the input/output structure comparison cost 0.61 in of height
for one idea.

### Constraints

Graphviz routes the edges, but it does *not* choose the canvas size: every band is
pinned to 460 pt by the `WIDTH` attributes in its leading spacer row — a 56 pt rail
plus 404 pt of content. The header comment in the DOT file explains why in detail.
In short:

- The figure is included at `width=1\linewidth` in a 6.5 in MDPI column, so a figure
  drawn wider is downscaled and its type shrinks with it
  (`printed_pt = font_pt * 6.5 / fig_width_in`). At 460 pt the 7.5 pt detail text
  prints at 7.56 pt, clearing the `MIN_PRINTED_PT = 7.0` floor in
  `src/utils/plotstyle.py`.
- **A cell holds text or a table, never both.** Mixing them is a syntax error, not a
  layout bug. This is why every compound box is a nested table rather than a heading
  followed by a pictogram.
- Graphviz never wraps text. Lines break only at `<BR/>`, so a long line silently
  widens its column and breaks the width constraint.
- Band heights are usually set by a *pictogram or a label cell*, not by the prose, so
  trimming text often changes nothing. Measure before and after: during development a
  whole stage was pinned by the stacked letters of a vertical label, and a global font
  reduction across the entire figure bought 3 pt.
- `BORDER="1"` adds to a cell *beyond* its declared `WIDTH`, so a band whose columns
  sum to exactly 460 pt renders a few points wider. Shave the declared widths until
  the measured width comes back. This bites twice over on nested tables: stage 4 ran
  4 pt wide purely from the borders on its two lane cells, with no text involved.
  Before hunting for a long line, copy the file and blank each block's text in turn —
  if the width does not move, the cause is structural.
- Global `sed` on a `WIDTH="nn"` value will hit every band that happens to share that
  number. Narrowing the stage 4 model boxes also narrowed the stage 1 and 2 source
  columns, which left a visibly ragged right edge. Check every band's measured width
  after any width edit, not just the one being fixed.
- Edges aimed at a *port* rather than at the node re-centre that node: pointing the
  stage 3 to stage 4 edges at the two lane ports shifted stage 4 sideways by 0.44 in
  and widened the whole figure. Keep the inter-stage edges node-to-node and let
  adjacency and colour carry the fork.
- Vertical alignment across two sibling cells is not predictable from padding
  arithmetic. The stage 4 lane headers sat 3.8 pt apart despite identical `CELLPADDING`
  on both, because a leading spacer row and a two-line cell each contribute leading
  that a hand calculation misses. Measure it instead: render at 150 dpi, take the node
  box from `dot -Tplain`, and scan each header's x-band for its first dark row. Two
  iterations got it to half a point.

**After editing, re-render and confirm the reported width is still about 6.396.** The
render script checks this and warns if a line has overrun its column, or if the height
has gone over the 5.5 in budget. The figure is currently 6.389 x 5.46 in.

### Rendering

The script resolves paths relative to itself, so it can be invoked from anywhere.
From the repository root (where the VS Code integrated terminal opens):

```powershell
.\docs\figures\render_wq_pipeline.ps1
```

Note that `.\render_wq_pipeline.ps1` fails from the repository root — the script is
not in that directory. In Git Bash, invoke it as
`powershell.exe -File ./docs/figures/render_wq_pipeline.ps1` instead.

The script locates Graphviz by first checking for `dot` on PATH, then falling back to
`C:\Program Files\Graphviz\bin\dot.exe`. On the current development machine Graphviz
is *not* on PATH, so the fallback is what actually runs. If you install Graphviz
somewhere else, or to a versioned directory, either add its `bin` to PATH or update
the fallback in the script.

The checked-in PNG is only a generated artifact. Re-run the script after editing the
DOT source and include the refreshed draft PNG in the submission zip.

## Superseded

`wq_modelling_workflow.dot`, `render_wq_modelling_workflow.ps1` and their PNGs are the
previous modeling-workflow figure, retained only until the replacement is signed off.
Delete them once that is done.
