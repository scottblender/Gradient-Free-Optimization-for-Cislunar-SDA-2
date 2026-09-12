# Manuscript figure style and export pipeline

All manuscript figures use one final formatting and export contract so that figures remain readable after being reduced in LaTeX while retaining consistent outer dimensions.

## Common canvas

Every manuscript figure is exported on the same `6.5 x 5.2 in` canvas. EPS output uses the painters renderer with `-depsc2 -painters -r600 -loose`, which preserves the common paper bounding box instead of tightening the EPS around the visible content. LaTeX should therefore size figures by width only; no per-panel `height`, `trim`, or `clip` adjustment is required.

## Typography

The centralized values in `scripts/reviewer2_paper_style.m` are intentionally larger than ordinary screen defaults because the full-size EPS panels are reduced when they are placed in two- or three-column manuscript grids.

- 2-D tick text: 18 pt
- 2-D axis labels: 21 pt
- 2-D legends: 16 pt
- 3-D tick text: 20 pt
- 3-D axis labels: 22 pt
- 3-D legends: 18 pt

Plot-specific annotations can retain deliberately chosen sizes when scientific readability requires them.

## Final fit-to-canvas pass

Immediately before EPS export, `scripts/finalize_manuscript_figure.m` applies the common typography and calls `scripts/center_manuscript_content.m` for single-panel exports. The outer canvas is never resized. If the larger text would extend beyond the canvas, only the inner axes rectangle is reduced until the complete axes/tick/label/legend extent fits inside a small safety margin. The visible content is then centered horizontally and vertically.

This final pass applies to both 2-D and 3-D manuscript figures. Composite MATLAB figures retain their custom internal layout; the manuscript generally exports individual panels and assembles grids in LaTeX.

## Three-dimensional figures

All 3-D manuscript panels use the same final typography, fit, centering, canvas, and EPS writer. Scientific view choices remain plot-specific: camera angle, projection, axis limits, and trajectory data are not homogenized when they carry physical meaning. Thus LG, LT, GI, orbit-family, baseline, and optimizer-comparison geometries share the same presentation pipeline while retaining their intended views.

For representative result geometries, panel-specific data limits prevent a wide orbit from another configuration from compressing the visible content of the current panel. Equal data aspect and plot-box aspect settings preserve the intended three-dimensional geometry.

## LaTeX usage

Use the exported figure at the width required by the manuscript layout, for example:

```latex
\includegraphics[width=\linewidth]{Fig7_1.eps}
```

For aligned subfigures, keep the existing top-aligned structure:

```latex
\begin{subfigure}[t]{0.32\textwidth}
    \vspace{0pt}
    \centering
    \includegraphics[width=\linewidth]{Fig7_1.eps}
    \caption{LG, 3 observers}
\end{subfigure}
```

The common EPS bounding box and final fit pass are intended to make manual LaTeX cropping unnecessary.
