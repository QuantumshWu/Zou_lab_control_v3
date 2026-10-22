# Live compose vs matplotlib's full draw

zlc-figure reproduces matplotlib's full draw of a materialized figure (the
`save()` path) call for call. The product oracle, though, is the live FRONT,
which `MatplotlibRenderer._compose_frame` paints from a cached Agg chrome
background plus zlc's native kernels. Where the two differ, the native
renderer must do what the compose does. This records where they differ and
why, measured on recbase (450 fronts) and the golden corpus (1584 document
fronts; pulse scenarios excluded).

Tools (runs dir `figure_oracle/`): `compose_kernels.py` wraps the painting
kernels (`zlc_plot._raster_kernels.raster_*`, `zlc_python.paint_masks`) and the
text-raster memo, attributes every call to a panel by the canvas buffer it
writes, and redraws each differing front's materialized figure with
`raster_error_bars`/`raster_polylines` substituted for Agg; `ck_cover.py`
names the artists whose footprint covers what is still different;
`ck_spines.py` tests the spines-under-data hypothesis.

## Equal / differ

| kind | recbase equal | corpus equal | corpus differ, explained by kernels |
|---|---|---|---|
| Image | 117/117 | 293/504 | 0 |
| Curve | 109/110 | 130/422 | 39 (+ spine order, below) |
| Facet | 1/111 | 92/220 | 83 |
| Rolling | 0/112 | 31/214 | 133 |
| Histogram | – | 212/224 | 10 |

## What paints the front

Always Agg: the chrome background (patch, spines, ticks, labels, titles),
restored per frame with `restore_region`. Kernels then paint the data:

* `raster_prepared_images` (zlc-render `composition.rs`): every Image surface
  (Image panels, image Facet cells). Equal to matplotlib's `_make_image` +
  `_image.resample` NEAREST in recbase, but NOT for some scalings: the
  corpus' small images (60x80 into ~190x250 px: `image_area_selector`,
  `image_color_limits`, `image_colorbar_side_distribution`, `dpr_*`,
  `state_*`, ...) differ on whole rows in a regular stripe pattern (a row
  sampling phase difference between the kernel's DDA and matplotlib's span
  interpolator; the exact rounding point is not pinned down). Seen in every
  sampled differing corpus Image front (maxdiff up to 234).
* `raster_polylines` (+ `transform_polylines`): Curve/Rolling/Facet data and
  fit lines. Not Agg's stroke (width floored at 1 px, its own coverage and
  caps).
* `raster_error_bars`: `_SegmentBuffered` error bars (Curve, Rolling, Facet).
  Not Agg's `draw_path_collection` of the materialized bar polygons.
* `raster_histogram_bars`: histogram bars. Equal to Agg wherever measured.
* `raster_fit_ellipses`: Facet image-cell fit ellipses (not exercised by a
  differing front here).
* `paint_masks`: the Facet chrome strokes (marks, frames) replayed as Agg
  coverage masks, and foreground text masks. Equal to Agg.
* text raster memo (`_install_text_raster_memo`): a string's Agg raster blitted
  back at a new position. Equal to Agg wherever measured.

## Why they differ

1. **Error-bar and polyline kernels** -- substituting them for Agg in the full
   draw makes the full draw equal the front for 110/110 differing recbase
   Facet fronts, 36/112 Rolling, and in the corpus 83 Facet, 133 Rolling,
   39 Curve, 10 Histogram fronts.
2. **Curve: spines under the data.** The compose bakes the spines (zorder
   2.5, above the data in a full draw) into the background and strokes the
   data over them. With the kernels substituted AND the spines drawn before
   the data, 29 of 31 sampled unexplained Curve fronts become equal
   (maxdiff 102-114 where a data line crosses a frame side).
3. **Image row phase** in `raster_prepared_images` (above): 0 of the 211
   differing corpus Image fronts are explained by anything else.
4. **Rolling residual**: 76/112 recbase Rolling fronts still differ by 1-2
   levels in a small region at the newest samples after (1); the region
   moves with the window. Neither text stacking nor spine order explains it;
   the prepared-curve command strokes geometry slightly different from the
   materialized Line2D there (unresolved).
5. Smaller unexplained classes in the corpus (one sampled front each):
   Facet histogram cells (bars + text, maxdiff 175), Facet image cells
   (image + frame), a Histogram fit with mathtext, a Rolling front where a
   spine differs by 255.

## Consequence for the native renderer

zlc-figure's Agg-call stream is the reference for every artist the compose
draws with Agg (chrome, text, markers, patches, collections, mesh,
histogram bars, masks). For the kernel-painted layers the front is the
kernel's output: the native renderer must paint Image surfaces, data and fit
polylines and error bars with the zlc-render kernels (same inputs the
`_*_plan` builders compute), in the compose's stacking (Curve: data over the
spines), not through `draw_path`/`draw_image`.
