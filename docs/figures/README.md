# Figure Sources

## Modeling workflow

`wq_modelling_workflow.dot` is the editable source for the modeling-pipeline
figure. Graphviz routes the edges, but it does *not* choose the canvas size: each
of the five stage bands is pinned to 460 pt (6.389 in) by the `WIDTH` attributes
in its leading spacer row. That is deliberate, and the header comment in the DOT
file explains why in detail. In short:

- The figure is included at `width=1\linewidth` in a 6.5 in MDPI column, so a
  figure drawn wider is downscaled and its type shrinks with it
  (`printed_pt = font_pt * 6.5 / fig_width_in`). An earlier version was 14 in
  wide with 9 pt text and printed at 4.2 pt. At 460 pt the figure is scaled
  *up* slightly and the 7.5 pt detail text prints at 7.56 pt, clearing the
  `MIN_PRINTED_PT = 7.0` floor in `src/utils/plotstyle.py`.
- Graphviz never wraps text. Lines break only at `<BR/>`, so a long line silently
  widens its column and breaks the width constraint.
- Bands 1 and 2 must keep identical column widths, or the three fan-in edges
  between them stop being vertical and `splines=ortho` routes them as steps.

It is a workflow graphic, not a written summary of the method: cells carry noun
phrases and numbers, and one detail line wherever the line fits its column. Full
sentences, equations, and instrument detail belong in the running text and in the
caption. A previous revision put them here and the figure grew to a full page.
The figure is currently 6.389 x 3.63 in.

**After editing, re-render and confirm `dot -Tplain` still reports a width of
6.3889.** If it grew, a line is too long for its column.

Change nodes and directed edges in the DOT file, then run the render script.

The script resolves paths relative to itself, so it can be invoked from anywhere.
From the repository root (where the VS Code integrated terminal opens):

```powershell
.\docs\figures\render_wq_modelling_workflow.ps1
```

From this directory:

```powershell
.\render_wq_modelling_workflow.ps1
```

Note that `.\render_wq_modelling_workflow.ps1` fails from the repository root —
the script is not in that directory. In Git Bash, invoke it as
`powershell.exe -File ./docs/figures/render_wq_modelling_workflow.ps1` instead.

The script locates Graphviz by first checking for `dot` on PATH, then falling
back to `C:\Program Files\Graphviz\bin\dot.exe`. On the current development
machine Graphviz is *not* on PATH, so the fallback is what actually runs. If you
install Graphviz somewhere else, or to a versioned directory, either add its
`bin` to PATH or update the fallback in the script.

It renders through the Cairo backend at 600 dpi and writes two identical PNG files:

- `docs/figures/wq_modelling_workflow.png` for repository use.
- `docs/report/draft/figures/wq_modelling_workflow.png` for the self-contained
  LaTeX submission bundle.

The checked-in PNG is only the generated LaTeX artifact. Re-run the command after
editing the DOT source and include the refreshed draft PNG in the submission zip.
