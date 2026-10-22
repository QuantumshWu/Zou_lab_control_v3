//! The live front (`_compose_frame` with the native kernels engaged): the
//! chrome background drawn by zlc-figure with every dynamic artist withheld,
//! then the dynamic artists in full-draw stacking, with the series data and
//! error bars painted by zlc-render's stroke kernels at their place in it.
//!
//! What is dynamic follows `_dynamic_artists`: the renderer's own artists,
//! plus -- for every axes one of them visibly stands on -- the axes' boundary
//! chrome (tick marks, spines) and the titles above it. An axes whose data is
//! only a kernel scene keeps its chrome in the background, under the data.
use std::sync::Arc;

use zlc_figure::agg::AggBackend;
use zlc_figure::axis::{tick_artists, ticks_to_draw};
use zlc_figure::draw::{draw_artist, Ctx};
use super::fonts::Math;
use super::foreground::{Lower, Lowering};
use zlc_figure::ir::{self, Artist};
use zlc_render::composition::{raster_histogram_bars, HistogramPlan};
use zlc_render::strokes::{raster_error_bars, raster_polylines, transform_polylines, ErrorBarBatch, PolylineBatch};
use zlc_render::Canvas;

use super::core::Renderer;
use super::curve::{effective_cap_width, series_slot, PreparedSeries};
use super::envelope::{bounded_isolated_points, thinned_to_columns};
use super::mpl::axes::{Child, ChildId};
use super::mpl::color;
use crate::configuration::spec::Surface;
use crate::projection::output::Payload;

/// One entry of the compose's dynamic stacking.
#[derive(Clone, Debug)]
enum Item {
    Child(ChildId),
    TickLines(bool),
    Spine(usize),
    Title(usize),
    Axis(bool),
    /// A grid cell's chrome stroke (its index among the cell's strokes).
    Stroke(usize),
    /// A figure child (its position in the IR's children): a grid's cell title.
    Figure(usize),
}

#[derive(Clone, Debug)]
struct Entry { axes: usize, zorder: f64, seq: usize, item: Item }

/// A finished front: RGBA rows of the raster size, shared (not copied) with
/// whoever shows it; its buffer returns to the renderer's pool when released.
pub struct Front { pub width: usize, pub height: usize, pub pixels: zlc_core::OwnedBuffer<u8> }

/// The canvas a compose left under an image's point overlay -- the
/// background, the picture, its forward chrome and the dynamics below the
/// rings -- with the key of all it drew and the data it drew it from (held,
/// so the key's identity of it stays unique). A frame that changes only
/// what stands from the rings up (a review's marks, a layer's statuses over
/// the same picture) paints that over a copy of it.
#[derive(Clone)]
pub struct Under { key: u128, payload: Arc<Payload>, pixels: Arc<Vec<u8>> }

/// Agg's cleared canvas colour (`RendererAgg::clear`): transparent white.
const CLEAR: [u8; 4] = [255, 255, 255, 0];

fn clip_of(bbox: [f64; 4], width: usize, height: usize) -> [i32; 4] {
    let h = height as f64;
    [
        0.max(bbox[0].floor() as i64) as i32,
        0.max((h - bbox[3]).floor() as i64) as i32,
        (width as i64).min(bbox[2].ceil() as i64) as i32,
        (height as i64).min((h - bbox[1]).ceil() as i64) as i32,
    ]
}

/// `np.clip(np.rint(rgba * 255), 0, 255).astype(uint8)`.
fn packed(rgba: [f64; 4]) -> [u8; 4] { rgba.map(|v| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u8) }

/// Every pixel of `pixels` set to `rgba`.
fn fill_rgba(pixels: &mut [u8], rgba: [u8; 4]) {
    if pixels.as_ptr().align_offset(4) != 0 {
        for pixel in pixels.chunks_exact_mut(4) { pixel.copy_from_slice(&rgba); }
        return;
    }
    // SAFETY: the start is word aligned (checked) and any four bytes are a valid u32.
    let (_, words, _) = unsafe { pixels.align_to_mut::<u32>() };
    words.fill(u32::from_ne_bytes(rgba));
}

fn child_index(children: &[(ChildId, Child)], id: ChildId) -> Option<usize> { children.iter().position(|(c, _)| *c == id) }

fn child_visible(child: &Child) -> bool {
    child.visible()
}

fn child_zorder(child: &Child) -> f64 {
    child.zorder()
}

impl Renderer {
    /// The zorder `axes`' Axis artists draw at.
    fn axis_zorder(&self, axes: usize) -> f64 { self.fig.axes[axes].axis_zorder.unwrap_or(self.rc.axis_zorder) }

    /// The dynamic artists (`_dynamic_artists`) in insertion order, and the axes that own chrome.
    fn dynamic_entries(&self) -> Vec<Entry> {
        let mut entries: Vec<Entry> = Vec::new();
        let push = |entries: &mut Vec<Entry>, axes: usize, zorder: f64, item: Item| {
            let seq = entries.len();
            entries.push(Entry { axes, zorder, seq, item });
        };
        // `_artists` in insertion order: series lines by key, the rolling
        // meter, the distribution and its axes, inspector annotations.
        for (_, (axes, lines)) in &self.curve.lines {
            for &line in lines { push(&mut entries, *axes, child_zorder(self.fig.axes[*axes].child(line).unwrap()), Item::Child(line)); }
        }
        if let Some((axes, id)) = self.rolling.latest {
            push(&mut entries, axes, child_zorder(self.fig.axes[axes].child(id).unwrap()), Item::Child(id));
        }
        if let Some((axes, id)) = self.rolling.distribution {
            push(&mut entries, axes, child_zorder(self.fig.axes[axes].child(id).unwrap()), Item::Child(id));
            let z = self.axis_zorder(axes);
            push(&mut entries, axes, z, Item::Axis(true));
            push(&mut entries, axes, z, Item::Axis(false));
        }
        // The image surface: the picture, the rail with its axes and guides, and the
        // colorbar's end fills, ramp, dividers, outline and long axis.
        let child = |entries: &mut Vec<Entry>, axes: usize, id: ChildId| {
            let z = child_zorder(self.fig.axes[axes].child(id).unwrap());
            push(entries, axes, z, Item::Child(id));
        };
        if let Some((axes, id)) = self.image.image { child(&mut entries, axes, id); }
        if let (Some((axes, id)), _) = self.image.distribution {
            child(&mut entries, axes, id);
            push(&mut entries, axes, self.rc.axis_zorder, Item::Axis(true));
            push(&mut entries, axes, self.rc.axis_zorder, Item::Axis(false));
        }
        if let Some((axes, guides)) = self.image.guides { for id in guides { child(&mut entries, axes, id); } }
        if let Some(colorbar) = &self.image.colorbar {
            let axes = colorbar.axes;
            for id in colorbar.ends.into_iter().chain([colorbar.solids, colorbar.dividers]) { child(&mut entries, axes, id); }
            push(&mut entries, axes, 2.5, Item::Spine(4));
            push(&mut entries, axes, self.rc.axis_zorder, Item::Axis(false));
        }
        // The height-bar scene's chrome (axis lines, grid, labels), its cage and readout.
        if let Some((axes, grid, lines, texts)) = &self.height_bars.chrome {
            for &id in [lines, grid].into_iter().chain(texts) { child(&mut entries, *axes, id); }
        }
        for (axes, id) in [self.height_bars.cage, self.height_bars.readout].into_iter().flatten() { child(&mut entries, axes, id); }
        // The point overlay: its label lists, then its rings, paths and timebase as made.
        // A grid's other image cells hold their overlays in their own tables.
        for points in std::iter::once(&self.image.points).chain(self.facet.images.values().map(|tables| &tables.points)) {
            if !points.listed { continue; }
            let mut made: Vec<ChildId> = points.rings.into_iter().chain(points.joined).chain(points.timebase).collect();
            made.sort_unstable();
            for &id in points.labels.iter().chain(&points.path_labels).chain(&made) { child(&mut entries, points.axes, id); }
        }
        for (axes, id) in self.histogram.bars() {
            push(&mut entries, axes, child_zorder(self.fig.axes[axes].child(id).unwrap()), Item::Child(id));
        }
        for (&axes, &id) in &self.curve.annotations {
            push(&mut entries, axes, child_zorder(self.fig.axes[axes].child(id).unwrap()), Item::Child(id));
        }
        // A pulse timeline's artists, group by group (`_artists` order).
        for (axes, id) in self.pulse.artists() {
            push(&mut entries, axes, child_zorder(self.fig.axes[axes].child(id).unwrap()), Item::Child(id));
        }
        // A view drag's axes: its ticks and labels move with the view.
        if let Some(confined) = self.confined {
            if self.fig.axes[confined].axison {
                let z = self.axis_zorder(confined);
                push(&mut entries, confined, z, Item::Axis(true));
                push(&mut entries, confined, z, Item::Axis(false));
            }
        }
        for (&axes, bars) in &self.curve.series_bars {
            for (_, ids) in bars {
                for &id in ids {
                    if let Some(child) = self.fig.axes[axes].child(id) { push(&mut entries, axes, child_zorder(child), Item::Child(id)); }
                }
            }
        }
        // The overlay layers' artists: selectors kind by kind, the fit's,
        // the source scatter, then any other layer's, in their order.
        let overlays: Vec<(usize, ChildId)> = self.selectors.artists().chain(self.fit.artists.iter().copied())
            .chain(self.classifier.artists()).chain(self.fit.source_scatter).chain(self.overlay_artists.iter().copied()).collect();
        for &(axes, id) in &overlays {
            if let Some(child) = self.fig.axes[axes].child(id) { push(&mut entries, axes, child_zorder(child), Item::Child(id)); }
        }
        // During a drag only what can change is dynamic: artists on axes the
        // dragged one does not meet stay in the background. Hidden axes
        // contribute nothing, as in a full draw.
        let size = (self.fig.size_inches, self.fig.dpi);
        let touchable = |axes: usize| match self.confined {
            None => true,
            Some(confined) => axes == confined || super::geometry::boxes_meet(
                self.fig.axes[axes].bbox_exact(size.0, size.1), self.fig.axes[confined].bbox_exact(size.0, size.1)),
        };
        entries.retain(|e| self.fig.axes[e.axes].visible && touchable(e.axes));
        for (seq, entry) in entries.iter_mut().enumerate() { entry.seq = seq; }
        // Boundary chrome of every axes a visible dynamic artist stands on.
        let mut owners: Vec<usize> = Vec::new();
        for entry in &entries {
            let visible = match entry.item {
                Item::Child(id) => self.fig.axes[entry.axes].child(id).is_some_and(child_visible),
                Item::Axis(_) => true,
                _ => false,
            };
            if visible && !owners.contains(&entry.axes) { owners.push(entry.axes); }
        }
        for axes in owners {
            let model = &self.fig.axes[axes];
            if !model.visible || !model.axison { continue; }
            let mut chrome: Vec<(f64, Item)> = Vec::new();
            for x in [true, false] {
                // A dynamic Axis draws its own tick marks.
                if entries.iter().any(|e| e.axes == axes && matches!(e.item, Item::Axis(a) if a == x)) { continue; }
                let p = &model.axis(x).major;
                if p.tick1on || p.tick2on || p.grid_on { chrome.push((self.axis_zorder(axes), Item::TickLines(x))); }
            }
            for k in 0..4 { chrome.push((2.5, Item::Spine(k))); }
            let lowest = chrome.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
            if !chrome.is_empty() && !model.titles[0].text.is_empty() && 3.0 > lowest { chrome.push((3.0, Item::Title(0))); }
            for (zorder, item) in chrome { push(&mut entries, axes, zorder, item); }
        }
        // A grid's chrome is always composed: each cell's marks and frame on
        // the cell, its title after every axes (a figure child).
        if let Some(ids) = &self.facet.chrome {
            for index in 0..self.fig.axes.len() {
                let model = &self.fig.axes[index];
                if !model.visible || !touchable(index) { continue; }
                for (k, stroke) in model.strokes.iter().enumerate() { push(&mut entries, index, stroke.zorder(), Item::Stroke(k)); }
            }
            for &id in ids {
                let Some(position) = self.fig.child_position(id) else { continue };
                if let Some(super::mpl::figure::FigureArtist::Text(title)) = self.fig.artists.get(position).map(|(_, a)| a) {
                    if title.visible { push(&mut entries, self.fig.axes.len(), title.zorder, Item::Figure(position)); }
                }
            }
        }
        entries
    }

    /// What a compose paints under a single image's point overlay: the key of
    /// all of it -- the background, the canvas size, the picture (its data,
    /// colouring and place) and every entry the stacking (`ordered`) paints
    /// before the overlay's first -- and that entry's `seq`. None for a grid,
    /// a colour drag, painted bars or an image without the overlay.
    fn under_plan(&self, figure: &ir::Figure, ordered: &[Entry], background: u64, lowered: bool, size: (u32, u32), bars: bool) -> Option<(u128, usize)> {
        use std::hash::Hasher;
        if bars || self.is_facet() || self.image.color_gesture { return None; }
        let (prepared, (axes, image), payload) = (self.image.prepared.as_ref()?, self.image.image?, self.last_payload.as_ref()?);
        let points = &self.image.points;
        if !points.listed || points.axes != axes { return None; }
        let layer: Vec<ChildId> = points.rings.into_iter().chain(points.joined).chain(points.timebase)
            .chain(points.labels.iter().copied()).chain(points.path_labels.iter().copied()).collect();
        let forward = |e: &Entry| e.axes == axes && matches!(e.item, Item::TickLines(_) | Item::Spine(_) | Item::Title(_));
        let picture = |e: &Entry| e.axes == axes && matches!(e.item, Item::Child(id) if id == image);
        let rest: Vec<&Entry> = ordered.iter().filter(|e| !forward(e) && !picture(e)).collect();
        let first = rest.iter().position(|e| e.axes == axes && matches!(e.item, Item::Child(id) if layer.contains(&id)))?;
        let nodes = super::hashkey::node_hashes(figure);
        let mut key = super::hashkey::Lanes::new();
        key.write_u64(background);
        key.write_u32(size.0);
        key.write_u32(size.1);
        key.f64(figure.dpi);
        key.write_u8(u8::from(lowered));
        key.write_usize(Arc::as_ptr(payload) as *const () as usize);
        key.write_u64(self.data_revision.unwrap_or(u64::MAX));
        key.write(prepared.key.as_bytes());
        key.f64s(&prepared.extent);
        key.f64(prepared.color_limits.0);
        key.f64(prepared.color_limits.1);
        key.write(prepared.cmap.as_bytes());
        key.write_u8(u8::from(matches!(prepared.state.get("show_image"), Some(crate::configuration::Value::Bool(true)))));
        for entry in ordered.iter().filter(|e| forward(e) || picture(e)).chain(rest[..first].iter().copied()) {
            let held = &figure.axes[entry.axes];
            key.write_usize(entry.axes);
            key.f64(entry.zorder);
            key.write_u64(super::hashkey::keyed(&nodes, &(held.position_original, held.position_active, held.view_lim, held.axison, held.frameon)));
            key.write_u64(match entry.item {
                Item::Child(id) => super::hashkey::keyed(&nodes, &held.children[child_index(&self.fig.axes[entry.axes].children, id)?]),
                Item::TickLines(x) => super::hashkey::keyed(&nodes, &(0u8, x, if x { &held.xaxis } else { &held.yaxis })),
                Item::Axis(x) => super::hashkey::keyed(&nodes, &(1u8, x, if x { &held.xaxis } else { &held.yaxis })),
                Item::Spine(k) => super::hashkey::keyed(&nodes, &(2u8, &held.spines[k])),
                Item::Title(k) => super::hashkey::keyed(&nodes, &(3u8, &held.titles[k])),
                Item::Stroke(_) | Item::Figure(_) => return None,
            });
        }
        Some((key.finish128(), rest[first].seq))
    }

    /// `_series_stacking_floor`.
    fn series_stacking_floor(&self) -> f64 {
        let mut zorders = Vec::new();
        for (&axes, records) in &self.curve.series_lines {
            for (line, _, _) in records { zorders.push(child_zorder(self.fig.axes[axes].child(*line).unwrap())); }
        }
        for (&axes, bars) in &self.curve.series_bars {
            for (_, ids) in bars { for &id in ids { if let Some(c) = self.fig.axes[axes].child(id) { zorders.push(child_zorder(c)); } } }
        }
        zorders.into_iter().fold(None, |m: Option<f64>, z| Some(m.map_or(z, |m| if z < m { z } else { m }))).unwrap_or(2.0)
    }

    /// The front of the current scene (`present` then `_compose_frame`), held as the renderer's front.
    /// A draw the fonts or the backend refuse is an error, never a panic.
    pub fn compose(&mut self) -> zlc_figure::Result<Arc<Front>> {
        let _all = super::profile::span("compose");
        let _kind = super::profile::span(match self.spec.kind().value() {
            "curve" => "compose[curve]", "image" => "compose[image]", "histogram" => "compose[histogram]",
            "rolling" => "compose[rolling]", "facet_grid" => "compose[facet_grid]", _ => "compose[pulse_timeline]",
        });
        let span = super::profile::span("compose.ir");
        // The draw's `_get_draw_artists`: every axes settles its aspect first.
        for index in 0..self.fig.axes.len() { self.apply_aspect(index); }
        self.update_ticks_for_draw();
        self.settle_path_labels();
        // Bars the kernel paints are never drawn from the IR: their
        // collections carry no paths there (only an export's IR makes them).
        let kernel_bars = self.kernel_bars();
        self.mark_kernel_bars(kernel_bars.as_deref().unwrap_or(&[]), true);
        let mut figure = self.ir();
        self.mark_kernel_bars(kernel_bars.as_deref().unwrap_or(&[]), false);
        let entries = self.dynamic_entries();
        drop(span);
        let span = super::profile::span("compose.background");
        // The background: the figure with every dynamic artist withheld (titles faded).
        // Withheld artists draw nothing, so they are neither drawn nor part of
        // what it is: the copy takes only the visible children that stay.
        let clone_span = super::profile::span("compose.background.clone");
        let mut withheld: Vec<Vec<bool>> = figure.axes.iter().map(|a| vec![false; a.children.len()]).collect();
        for entry in &entries {
            match entry.item {
                Item::Child(id) => {
                    if let Some(index) = child_index(&self.fig.axes[entry.axes].children, id) { withheld[entry.axes][index] = true; }
                }
                // A grid cell's chrome strokes follow the model's children in the IR.
                Item::Stroke(k) => withheld[entry.axes][self.fig.axes[entry.axes].children.len() + k] = true,
                _ => {}
            }
        }
        let children: Vec<Vec<Artist>> = figure.axes.iter_mut().map(|a| std::mem::take(&mut a.children)).collect();
        let mut background = figure.clone();
        for (((axes, copy), children), withheld) in figure.axes.iter_mut().zip(&mut background.axes).zip(children).zip(&withheld) {
            copy.children = children.iter().zip(withheld)
                .filter(|(child, withheld)| !**withheld && child.props().is_none_or(|p| p.visible))
                .map(|(child, _)| child.clone()).collect();
            axes.children = children;
        }
        for entry in &entries {
            if let Item::Figure(position) = entry.item {
                set_artist_visible(&mut background.children[position], false);
                continue;
            }
            let axes = &mut background.axes[entry.axes];
            match entry.item {
                Item::Child(_) | Item::Stroke(_) => {}
                Item::Figure(_) => unreachable!(),
                Item::TickLines(x) => {
                    let axis = if x { &mut axes.xaxis } else { &mut axes.yaxis };
                    // Every tick's own template too: a grid cell's ticks are not alike.
                    if let Some(major) = axis.major.as_mut() {
                        for tick in major.tick.iter_mut().chain(major.per_tick.iter_mut().flatten()) {
                            tick.tick1line.artist.visible = false;
                            tick.tick2line.artist.visible = false;
                            tick.gridline.artist.visible = false;
                        }
                    }
                }
                Item::Spine(k) => axes.spines[k].patch.artist.visible = false,
                Item::Title(k) => axes.titles[k].artist.alpha = Some(0.0),
                // A withheld Axis draws nothing: its ticks are no part of the background.
                Item::Axis(x) => {
                    let axis = if x { &mut axes.xaxis } else { &mut axes.yaxis };
                    axis.visible = false;
                    axis.major = None;
                    axis.minor = None;
                    axis.offset_string = None;
                }
            }
        }
        let (fw, fh) = self.fig.size_px();
        let dpi = self.fig.dpi;
        // The background is a pure function of what it draws: an unchanged
        // one is copied back instead of drawn.
        for axes in &mut background.axes {
            // An axes whose artists, axes and spines are all withheld draws only its patch.
            if axes.children.is_empty() && !axes.xaxis.visible && !axes.yaxis.visible && axes.spines.iter().all(|s| !s.patch.artist.visible) {
                axes.spines.clear();
                axes.view_lim = [0.0, 0.0, 1.0, 1.0];
            }
        }
        drop(clone_span);
        let key_span = super::profile::span("compose.background.key");
        let key = super::hashkey::figure_key(&background);
        drop(key_span);
        let (w, h) = (fw as u32, fh as u32);
        let size = w as usize * h as usize * 4;
        let mut ordered = entries.clone();
        ordered.sort_by(|a, b| (a.axes, a.zorder).partial_cmp(&(b.axes, b.zorder)).unwrap().then(a.seq.cmp(&b.seq)));
        // `_paint_foreground` lowers recorded boundary chrome to masks unless
        // a selector or confined view gesture holds the frame.
        let lowered = self.selector_gesture.is_none() && self.confined.is_none();
        // An image's point overlay over what it stood on last time starts from
        // that canvas; a re-present of the same data keeps it for the next one.
        let plan_span = super::profile::span("compose.under.plan");
        let under = self.under_plan(&figure, &ordered, key, lowered, (w, h), kernel_bars.is_some());
        drop(plan_span);
        let kept = under.and_then(|(wanted, _)| self.under.clone().filter(|under| under.key == wanted && under.pixels.len() == size
            && self.last_payload.as_ref().is_some_and(|payload| Arc::ptr_eq(payload, &under.payload))));
        let mut spare = if kept.is_some() { None } else { self.under.take().and_then(|under| Arc::try_unwrap(under.pixels).ok()) };
        let keep = under.filter(|_| kept.is_none() && self.payload_kept);
        let mut snapshot = None;
        // The canvas: the retained backend's own pixels when it kept them (a
        // front of another raster size), else the last front's buffer once its
        // readers released it, else a new allocation -- each written whole
        // below, so never zeroed first.
        self.front = None;
        let mut retained = self.canvas.0.take().filter(|b| (b.renderer.width, b.renderer.height, b.renderer.dpi) == (w, h, dpi));
        let mut pixels = retained.as_mut().map(|b| std::mem::take(&mut b.renderer.pixels)).filter(|p| p.len() == size)
            .or_else(|| self.front_pool.take_idle()).unwrap_or_default();
        let held = match &kept {
            Some(under) => Some(under.pixels.clone()),
            None => self.background.as_ref().filter(|(cached, store)| *cached == key && store.len() == size).map(|(_, p)| p.clone()),
        };
        let hit = held.is_some();
        // A drawn Axis keeps where its label landed (a hidden one shows it later).
        if !hit { self.record_draw_positions(&figure); }
        let mut fonts = self.fonts.lock().unwrap();
        let mut math = Math;
        match held {
            Some(held) => {
                let _copy = super::profile::span("compose.background.copy");
                pixels.clear();
                pixels.extend_from_slice(&held);
            }
            None => {
                let _clear = super::profile::span("compose.background.clear");
                if pixels.len() != size { pixels = vec![0; size]; }
                fill_rgba(&mut pixels, CLEAR);
            }
        }
        let mut backend = match retained {
            Some(mut backend) => { backend.renderer.pixels = pixels; backend }
            None => AggBackend::over_pixels(fw, fh, dpi, pixels).expect("canvas"),
        };
        if !hit {
            let _draw = super::profile::span("compose.background.draw");
            zlc_figure::draw(&background, &mut backend, &mut *fonts, &mut math)?;
            let mut store = self.background.take().and_then(|(_, store)| Arc::try_unwrap(store).ok()).unwrap_or_default();
            store.clear();
            store.extend_from_slice(&backend.renderer.pixels);
            self.background = Some((key, Arc::new(store)));
        }
        // A grid's chrome stroke or cell title is drawn from the calls it made
        // last time while what it draws is unchanged: the artist, the nodes it
        // reads (with the axes frames behind them) and the figure's size.
        let mut recordings = self.chrome_calls.0.take().unwrap_or_default();
        recordings.previous = std::mem::take(&mut recordings.next);
        let chrome_nodes = entries.iter().any(|e| matches!(e.item, Item::Stroke(_) | Item::Figure(_)))
            .then(|| super::hashkey::node_hashes(&figure));
        let chrome_key = |entry: &Entry| -> u64 {
            let nodes = chrome_nodes.as_deref().unwrap_or(&[]);
            let artist = match entry.item {
                Item::Stroke(k) => &figure.axes[entry.axes].children[self.fig.axes[entry.axes].children.len() + k],
                Item::Figure(position) => &figure.children[position],
                _ => unreachable!(),
            };
            super::hashkey::keyed(nodes, &(artist, figure.dpi, figure.size_inches))
        };
        drop(span);
        // Histogram bars are a prepared scene: painted first, under the
        // chrome they may cover; a scene the kernel cannot serve hands the
        // picture back to the collections.
        let native_bars = kernel_bars.inspect(|bars| self.raster_prepared_histograms(&mut backend, bars));
        if native_bars.is_none() { self.histogram.prepared = None; }
        let recorded = std::cell::RefCell::new(&mut recordings);
        let paint = |backend: &mut AggBackend, fonts: &mut super::fonts::FontSet, band: &[&Entry]| -> zlc_figure::Result<()> {
            if band.is_empty() { return Ok(()); }
            let _span = super::profile::span("compose.dynamics");
            let mut math = Math;
            let active = std::cell::Cell::new(Lower::Nothing);
            let mut memo = self.lowered.lock().unwrap();
            let mut lowering = Lowering { inner: backend, active: &active, memo: &mut memo };
            let recording = std::cell::RefCell::new(None);
            let mut tap = super::replay::Tap { inner: &mut lowering, recording: &recording };
            let ctx_span = super::profile::span("compose.dynamics.ctx");
            let mut ctx = Ctx::new(&figure, &mut tap, fonts, &mut math)?;
            for index in 0..figure.axes.len() { zlc_figure::axes::apply_aspect(&mut ctx, index)?; }
            drop(ctx_span);
            for entry in band {
                active.set(match entry.item {
                    Item::TickLines(_) if lowered => Lower::Markers,
                    Item::Spine(_) if lowered => Lower::Patch,
                    _ => Lower::Nothing,
                });
                let chrome = match entry.item {
                    Item::Stroke(k) => Some((entry.axes, k, false)),
                    Item::Figure(position) => Some((entry.axes, position, true)),
                    _ => None,
                };
                let Some(chrome) = chrome else {
                    self.draw_entry(&mut ctx, &figure, entry)?;
                    continue;
                };
                let key = chrome_key(entry);
                let mut recordings = recorded.borrow_mut();
                match recordings.previous.remove(&chrome).filter(|(recorded, _)| *recorded == key) {
                    Some((_, calls)) => {
                        super::replay::replay(ctx.backend, &calls)?;
                        recordings.next.insert(chrome, (key, calls));
                    }
                    None => {
                        *recording.borrow_mut() = Some(Vec::new());
                        self.draw_entry(&mut ctx, &figure, entry)?;
                        let calls = recording.borrow_mut().take().unwrap_or_default();
                        recordings.next.insert(chrome, (key, calls));
                    }
                }
            }
            Ok(())
        };
        // A prepared image is painted first; only the chrome it may have
        // covered -- its own axes' tick marks, spines and title -- comes
        // forward, then everything else in its full-draw stacking.
        let (w, h) = (backend.renderer.width as usize, backend.renderer.height as usize);
        let pictures = if kept.is_some() { Some(self.image.image.into_iter().collect()) } else {
            let _span = super::profile::span("compose.kernels");
            match self.raster_prepared_facet_images(&mut backend.renderer.pixels, w, h, &figure) {
                Some(pictures) => Some(pictures),
                None => self.raster_prepared_image(&mut backend.renderer.pixels, w, h, &figure).then(|| self.image.image.into_iter().collect()),
            }
        };
        if let Some(pictures) = pictures {
            // A grid's chrome comes forward with its cells' pictures.
            let image_axes: Vec<usize> = pictures.iter().map(|p| p.0).collect();
            let forward = |e: &&Entry| (image_axes.contains(&e.axes) && matches!(e.item, Item::TickLines(_) | Item::Spine(_) | Item::Title(_)))
                || matches!(e.item, Item::Stroke(_) | Item::Figure(_));
            let picture = |e: &&Entry| matches!(e.item, Item::Child(id) if pictures.contains(&(e.axes, id)));
            // A colour drag repaints, last, its own lines and everything else
            // on the picture's axes (`color_overlay`), over the rest.
            let colour: Vec<(usize, ChildId)> = if self.image.color_gesture {
                self.selectors.kinds.iter().filter(|(kind, _, _)| *kind == crate::interaction::scene::SceneKind::ColorLimits)
                    .flat_map(|(_, artists, _)| artists.iter().copied()).collect()
            } else { Vec::new() };
            let overlay = |e: &&Entry| !colour.is_empty() && !picture(e)
                && (image_axes.contains(&e.axes) || matches!(e.item, Item::Child(id) if colour.contains(&(e.axes, id))));
            let band: Vec<&Entry> = ordered.iter().filter(|e| forward(e) && !overlay(e)).collect();
            if kept.is_none() { paint(&mut backend, &mut fonts, &band)?; }
            // An overview's ellipse fits are the kernel's, over the forward chrome.
            let (w, h) = (backend.renderer.width as usize, backend.renderer.height as usize);
            let ellipses = if kept.is_some() { Vec::new() } else {
                let _span = super::profile::span("compose.kernels");
                self.raster_facet_fit_ellipses(&mut backend.renderer.pixels, w, h, &figure).unwrap_or_default()
            };
            let painted = |e: &&Entry| matches!(e.item, Item::Child(id) if ellipses.contains(&(e.axes, id)));
            let band: Vec<&Entry> = ordered.iter().filter(|e| !forward(e) && !picture(e) && !painted(e) && !overlay(e)).collect();
            // What stands under the point overlay is the kept canvas, or is kept now.
            match under.and_then(|(_, first)| band.iter().position(|e| e.seq == first)) {
                Some(at) if kept.is_some() => paint(&mut backend, &mut fonts, &band[at..])?,
                Some(at) if keep.is_some() => {
                    paint(&mut backend, &mut fonts, &band[..at])?;
                    let _copy = super::profile::span("compose.under.keep");
                    let mut store = spare.take().unwrap_or_default();
                    store.clear();
                    store.extend_from_slice(&backend.renderer.pixels);
                    snapshot = Some(store);
                    drop(_copy);
                    paint(&mut backend, &mut fonts, &band[at..])?;
                }
                _ => paint(&mut backend, &mut fonts, &band)?,
            }
            let band: Vec<&Entry> = ordered.iter().filter(overlay).collect();
            paint(&mut backend, &mut fonts, &band)?;
        } else if let Some(bars) = native_bars {
            // The bars stand at the bottom of their cells' stacking, already
            // painted; everything else follows in its own order, a fit drawn
            // over them stroked by the kernel at its place.
            let painted = |e: &&Entry| !matches!(e.item, Item::Child(id) if bars.contains(&(e.axes, id)));
            match self.histogram_fit_stroke() {
                Some(stroke) => {
                    let rest = |e: &&Entry| painted(e) && !matches!(e.item, Item::Child(id) if stroke.children.contains(&(e.axes, id)));
                    let under: Vec<&Entry> = ordered.iter().filter(rest).filter(|e| e.zorder < stroke.zorder).collect();
                    paint(&mut backend, &mut fonts, &under)?;
                    let (w, h) = (backend.renderer.width as usize, backend.renderer.height as usize);
                    {
                        let mut canvas = Canvas::new(&mut backend.renderer.pixels, w, h).expect("canvas");
                        let _span = super::profile::span("compose.kernels");
                        stroke.paint(&mut canvas);
                    }
                    let over: Vec<&Entry> = ordered.iter().filter(rest).filter(|e| e.zorder >= stroke.zorder).collect();
                    paint(&mut backend, &mut fonts, &over)?;
                }
                None => {
                    let all: Vec<&Entry> = ordered.iter().filter(painted).collect();
                    paint(&mut backend, &mut fonts, &all)?;
                }
            }
        } else {
            // What a full draw stacks below the series goes first; the kernels
            // stroke the series (and eligible fit lines) at their depths.
            let (floor, strokes) = match self.curve.prepared.clone() {
                Some(prepared) => {
                    let floor = self.series_stacking_floor();
                    let under: Vec<&Entry> = ordered.iter().filter(|e| e.zorder < floor).collect();
                    paint(&mut backend, &mut fonts, &under)?;
                    let span = super::profile::span("compose.kernels");
                    self.raster_prepared_curve(&mut backend, &figure, &mut fonts, &prepared.series)?;
                    drop(span);
                    (Some(floor), Some(self.native_strokes(true).unwrap_or_default()))
                }
                None => (None, self.native_strokes(false)),
            };
            match strokes {
                Some(strokes) => {
                    let native: Vec<(usize, ChildId)> = strokes.iter().flat_map(|s| s.children.iter().copied()).collect();
                    let rest = |e: &&Entry| !matches!(e.item, Item::Child(id) if native.contains(&(e.axes, id)));
                    let mut floor = floor;
                    for stroke in &strokes {
                        let band: Vec<&Entry> = ordered.iter().filter(rest)
                            .filter(|e| floor.is_none_or(|f| e.zorder >= f) && e.zorder < stroke.zorder).collect();
                        paint(&mut backend, &mut fonts, &band)?;
                        let (w, h) = (backend.renderer.width as usize, backend.renderer.height as usize);
                        let mut canvas = Canvas::new(&mut backend.renderer.pixels, w, h).expect("canvas");
                        let _span = super::profile::span("compose.kernels");
                        stroke.paint(&mut canvas);
                        floor = Some(stroke.zorder);
                    }
                    let band: Vec<&Entry> = ordered.iter().filter(rest).filter(|e| floor.is_none_or(|f| e.zorder >= f)).collect();
                    paint(&mut backend, &mut fonts, &band)?;
                }
                None => {
                    let all: Vec<&Entry> = ordered.iter().collect();
                    paint(&mut backend, &mut fonts, &all)?;
                }
            }
        }
        drop(fonts);
        // The chrome a kept canvas holds was not drawn now; its lowering stays.
        if kept.is_none() { self.lowered.lock().unwrap().sweep(); }
        drop(recorded);
        self.chrome_calls.0 = Some(recordings);
        if let (Some(pixels), Some((key, _)), Some(payload)) = (snapshot, keep, self.last_payload.clone()) {
            self.under = Some(Under { key, payload, pixels: Arc::new(pixels) });
        }
        self.raster_generation += 1;
        self.composed_generation = self.raster_generation as i64;
        let _span = super::profile::span("compose.publish");
        let front = Arc::new(self.published(&mut backend));
        self.canvas.0 = Some(backend);
        self.front = Some(front.clone());
        Ok(front)
    }

    /// `_published_rgba`: the canvas at the raster size the frontend contracts.
    /// A canvas of that size is the front itself (its buffer handed over);
    /// another is copied into one and the canvas keeps its own for the next compose.
    fn published(&mut self, backend: &mut AggBackend) -> Front {
        let (tw, th) = (self.plan.raster_size.0 as usize, self.plan.raster_size.1 as usize);
        let (aw, ah) = (backend.renderer.width as usize, backend.renderer.height as usize);
        let pixels = if (aw, ah) == (tw, th) {
            std::mem::take(&mut backend.renderer.pixels)
        } else {
            let mut pixels = self.front_pool.take_idle().unwrap_or_default();
            pixels.clear();
            pixels.resize(tw * th * 4, 0);
            fill_rgba(&mut pixels, packed(color::rgba(&self.rc.figure_facecolor)));
            for row in 0..th.min(ah) {
                let n = tw.min(aw) * 4;
                pixels[row * tw * 4..row * tw * 4 + n].copy_from_slice(&backend.renderer.pixels[row * aw * 4..row * aw * 4 + n]);
            }
            pixels
        };
        Front { width: tw, height: th, pixels: self.front_pool.share(pixels) }
    }

    fn draw_entry(&self, ctx: &mut Ctx<'_>, figure: &ir::Figure, entry: &Entry) -> zlc_figure::Result<()> {
        if let Item::Figure(position) = entry.item {
            return draw_artist(ctx, &figure.children[position]);
        }
        let axes = &figure.axes[entry.axes];
        let _span = super::profile::span(match entry.item {
            Item::Child(_) => "compose.dynamics.child",
            Item::TickLines(_) => "compose.dynamics.ticks",
            Item::Spine(_) => "compose.dynamics.spine",
            Item::Title(_) => "compose.dynamics.title",
            Item::Axis(_) => "compose.dynamics.axis",
            Item::Stroke(_) => "compose.dynamics.stroke",
            Item::Figure(_) => "compose.dynamics.figure",
        });
        match entry.item {
            Item::Stroke(k) => {
                let artist = &axes.children[self.fig.axes[entry.axes].children.len() + k];
                draw_artist(ctx, artist)?;
            }
            Item::Figure(_) => unreachable!(),
            Item::Child(id) => {
                let Some(index) = child_index(&self.fig.axes[entry.axes].children, id) else { return Ok(()) };
                let artist = &axes.children[index];
                let _kind = super::profile::span(match artist {
                    Artist::Line(_) => "compose.dynamics.child.line",
                    Artist::Text(_) => "compose.dynamics.child.text",
                    Artist::Patch(_) => "compose.dynamics.child.patch",
                    Artist::Collection(_) => "compose.dynamics.child.collection",
                    _ => "compose.dynamics.child.other",
                });
                if artist.props().is_some_and(|p| p.visible) { draw_artist(ctx, artist)?; }
            }
            Item::TickLines(x) => {
                let axis = if x { &axes.xaxis } else { &axes.yaxis };
                let ticks = ticks_to_draw(ctx, entry.axes, axis)?;
                for tick in &ticks {
                    let a = tick_artists(tick, x);
                    for line in [&a.gridline, &a.tick1line, &a.tick2line] {
                        if line.artist.visible { zlc_figure::lines::draw_line(ctx, line)?; }
                    }
                }
            }
            Item::Spine(k) => {
                let spine = &axes.spines[k];
                if spine.patch.artist.visible && axes.axison && axes.frameon {
                    zlc_figure::patches::draw_patch_in(ctx, &spine.patch, Some(entry.axes))?;
                }
            }
            Item::Title(k) => { zlc_figure::text::draw_text(ctx, &axes.titles[k], None)?; }
            Item::Axis(x) => {
                let axis = if x { &axes.xaxis } else { &axes.yaxis };
                if axis.visible { zlc_figure::axis::draw_axis(ctx, entry.axes, axis)?; }
            }
        }
        Ok(())
    }

    /// `_raster_prepared_curve_command` + `_raster_prepared_error_bars` (one surface).
    fn raster_prepared_curve(&self, backend: &mut AggBackend, figure: &ir::Figure, fonts: &mut super::fonts::FontSet,
                             series_by_cell: &[Vec<PreparedSeries>]) -> zlc_figure::Result<()> {
        let (width, height) = (backend.renderer.width as usize, backend.renderer.height as usize);
        let canvas_h = height as f64;
        let dpi = self.fig.dpi;
        let cycle = self.style.palette.line_cycle();
        let surfaces = self.painted_surfaces();
        // Error bars: one kernel launch for every cell.
        let policy = &self.style.render;
        let bar_width = (policy.uncertainty_bar_linewidth * dpi / 72.0).max(1.0);
        let cap_width = (2.0 * policy.uncertainty_bar_capsize_pt * dpi / 72.0).max(0.0);
        let (mut xs, mut lows, mut highs, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), vec![0i64]);
        let (mut colours, mut widths, mut caps, mut clips) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (&(_, axes, _), cell) in surfaces.iter().zip(series_by_cell) {
            let model = &self.fig.axes[axes];
            let bbox = model.bbox_exact(self.fig.size_inches, dpi);
            let clip = clip_of(bbox, width, height);
            let m = model.trans_data(self.fig.size_inches, dpi);
            let (a, b, c, d, e, f) = (m.0[0][0], m.0[1][0], m.0[0][1], m.0[1][1], m.0[0][2], m.0[1][2]);
            let band_items: Vec<&PreparedSeries> = cell.iter().filter(|item| item.band.is_some()).collect();
            // A non-affine transData (a scale kept from an image cell) takes
            // each point through `transform()`: its scales, then its affine.
            let linear = model.is_linear();
            let batched = linear && band_items.len() > 1 && band_items[1..].iter().all(|item| item.x.as_slice() == band_items[0].x.as_slice());
            let mut append = |item: &PreparedSeries, gx: Vec<f64>, low: Vec<f64>, high: Vec<f64>| {
                lows.extend(low.iter().zip(&high).map(|(l, h)| l.min(*h)).collect::<Vec<_>>());
                highs.extend(low.iter().zip(&high).map(|(l, h)| l.max(*h)).collect::<Vec<_>>());
                caps.push(effective_cap_width(&gx, cap_width));
                offsets.push(offsets.last().unwrap() + gx.len() as i64);
                xs.extend(gx);
                let mut rgba = color::rgba(cycle[series_slot(&item.group_key, cycle.len())]);
                rgba[3] *= policy.uncertainty_bar_alpha;
                colours.push(packed(rgba));
                widths.push(bar_width);
                clips.push(clip);
            };
            for item in &band_items {
                let (low, high) = item.band.as_ref().unwrap();
                if batched {
                    let valid: Vec<bool> = (0..item.x.len()).map(|p| item.valid[p] && low[p].is_finite() && high[p].is_finite() && high[p] > low[p]).collect();
                    if !valid.iter().any(|v| *v) { continue; }
                    let tr = |p: usize, y: f64| if valid[p] { (a * item.x[p] + c * y + e, canvas_h - (b * item.x[p] + d * y + f)) } else { (f64::NAN, f64::NAN) };
                    let gx: Vec<f64> = (0..item.x.len()).map(|p| tr(p, low[p]).0).collect();
                    let ly: Vec<f64> = (0..item.x.len()).map(|p| tr(p, low[p]).1).collect();
                    let hy: Vec<f64> = (0..item.x.len()).map(|p| tr(p, high[p]).1).collect();
                    append(item, gx, ly, hy);
                } else {
                    let usable: Vec<usize> = (0..item.x.len()).filter(|&p| item.valid[p] && low[p].is_finite() && high[p].is_finite() && high[p] > low[p]).collect();
                    if usable.is_empty() { continue; }
                    let low_points: Vec<(f64, f64)> = usable.iter().map(|&p| model.scaled(item.x[p], low[p])).collect();
                    let high_y: Vec<f64> = usable.iter().map(|&p| model.scaled(item.x[p], high[p]).1).collect();
                    let gx: Vec<f64> = low_points.iter().map(|&(x, low)| a * x + c * low + e).collect();
                    let ly: Vec<f64> = low_points.iter().map(|&(x, low)| canvas_h - (b * x + d * low + f)).collect();
                    let hy: Vec<f64> = low_points.iter().zip(&high_y).map(|(&(x, _), &high)| canvas_h - (b * x + d * high + f)).collect();
                    append(item, gx, ly, hy);
                }
            }
        }
        if !xs.is_empty() {
            let _span = super::profile::span("kernels.curve.bars");
            let mut canvas = Canvas::new(&mut backend.renderer.pixels, width, height).expect("canvas");
            raster_error_bars(&ErrorBarBatch { x: &xs, y_low: &lows, y_high: &highs, offsets: &offsets, colours: &colours,
                                               widths: &widths, cap_widths: &caps, clips: &clips }, &mut canvas).expect("error bars");
        }
        // Lines, per series order, each followed by its isolated glyphs.
        let line_policy = &self.style.artists.curve;
        let line_width = (line_policy.linewidth * dpi / 72.0).max(1.0);
        let mut vertices: Vec<Vec<[f64; 2]>> = Vec::new();
        let mut series_orders: Vec<Vec<usize>> = Vec::new();
        let mut singles: Vec<Option<(usize, Vec<f64>, Vec<f64>, usize)>> = Vec::new();
        let mut line_colours = Vec::new();
        let mut line_clips = Vec::new();
        for (&(_, axes, _), cell) in surfaces.iter().zip(series_by_cell) {
            let model = &self.fig.axes[axes];
            let bbox = model.bbox_exact(self.fig.size_inches, dpi);
            let clip = clip_of(bbox, width, height);
            let m = model.trans_data(self.fig.size_inches, dpi);
            let (a, b, c, d, e, f) = (m.0[0][0], m.0[1][0], m.0[0][1], m.0[1][1], m.0[0][2], m.0[1][2]);
            let (window, columns) = self.pixel_columns(axes);
            for (order, item) in cell.iter().enumerate() {
                let source_y: Vec<f64> = item.y.iter().zip(item.valid.iter()).map(|(&y, &v)| if v { y } else { f64::NAN }).collect();
                let (ix, iy) = thinned_to_columns(&item.x, &source_y, window, columns);
                vertices.push(ix.iter().zip(&iy).map(|(&x, &y)| {
                    let (x, y) = model.scaled(x, y);
                    [a * x + c * y + e, canvas_h - (b * x + d * y + f)]
                }).collect());
                let slot = series_slot(&item.group_key, cycle.len());
                let mut rgba = color::rgba(cycle[slot]);
                rgba[3] *= line_policy.alpha;
                line_colours.push(packed(rgba));
                line_clips.push(clip);
                if series_orders.len() <= order { series_orders.push(Vec::new()); }
                series_orders[order].push(vertices.len() - 1);
                let mut single = None;
                if item.summary.isolated {
                    let (px, py) = bounded_isolated_points(&item.x, &source_y, window, columns);
                    if !px.is_empty() { single = Some((axes, px, py, slot)); }
                }
                singles.push(single);
            }
        }
        if vertices.is_empty() { return Ok(()); }
        let passes: Vec<Vec<usize>> = if singles.iter().any(Option::is_some) { series_orders } else { vec![(0..vertices.len()).collect()] };
        let with_markers = singles.iter().any(Option::is_some);
        for indices in passes {
            let mut flat = Vec::new();
            let mut offs = vec![0i64];
            for &i in &indices { flat.extend_from_slice(&vertices[i]); offs.push(flat.len() as i64); }
            let colours: Vec<[u8; 4]> = indices.iter().map(|&i| line_colours[i]).collect();
            let widths = vec![line_width; indices.len()];
            let clips: Vec<[i32; 4]> = indices.iter().map(|&i| line_clips[i]).collect();
            {
                let _span = super::profile::span("kernels.curve.lines");
                let mut canvas = Canvas::new(&mut backend.renderer.pixels, width, height).expect("canvas");
                raster_polylines(&PolylineBatch { vertices: &flat, offsets: &offs, colours: &colours, widths: &widths, clips: &clips },
                                 &mut canvas).expect("polylines");
            }
            if with_markers {
                let _span = super::profile::span("kernels.curve.glyphs");
                for &i in &indices {
                    let Some((axes, px, py, slot)) = &singles[i] else { continue };
                    let line = self.glyph_marker(figure, *axes, px, py, cycle[*slot]);
                    let mut math = Math;
                    let mut ctx = Ctx::new(figure, backend, fonts, &mut math)?;
                    for index in 0..figure.axes.len() { zlc_figure::axes::apply_aspect(&mut ctx, index)?; }
                    zlc_figure::lines::draw_line(&mut ctx, &line)?;
                }
            }
        }
        Ok(())
    }

    /// The Line2D that draws a prepared scene's isolated glyphs.
    fn glyph_marker(&self, figure: &ir::Figure, axes: usize, x: &[f64], y: &[f64], colour: &str) -> ir::Line2D {
        let curve = &self.style.artists.curve;
        let spec = super::mpl::artist::LineSpec {
            color: colour, linewidth: self.rc.lines_linewidth, linestyle: "None", alpha: Some(curve.alpha), marker: "_",
            markersize: self.style.artists.curve_marker_size_pt, markeredgewidth: curve.linewidth, markeredgecolor: None,
            markerfacecolor: None, zorder: 2.0,
        };
        let a = &figure.axes[axes];
        let mut line = super::mpl::artist::line(&self.rc, &spec, x.iter().zip(y).map(|(&x, &y)| [x, y]).collect(), a.trans_data);
        line.artist.clip_box = Some(a.bbox);
        line
    }

    /// `_native_curve_lines` + the per-depth plans of `_compose_frame`'s
    /// stroke branch; with a prepared scene only the fit lines remain.
    fn native_strokes(&self, prepared: bool) -> Option<Vec<Stroke>> {
        let (bars, data, fit) = self.native_lines(prepared)?;
        let (fw, fh) = self.fig.size_px();
        let (width, height) = (fw as usize, fh as usize);
        let canvas_h = height as f64;
        let dpi = self.fig.dpi;
        let mut strokes: Vec<Stroke> = Vec::new();
        // Bars, then data lines, each batched by depth in first-seen order.
        let mut groups: Vec<(f64, Vec<(usize, ChildId)>, bool)> = Vec::new();
        let (bars, data) = if prepared { (Vec::new(), Vec::new()) } else { (bars, data) };
        for (list, is_bar) in [(&bars, true), (&data, false), (&fit, false)] {
            let mut by_depth: Vec<(f64, Vec<(usize, ChildId)>)> = Vec::new();
            for &(axes, id) in list.iter() {
                let z = child_zorder(self.fig.axes[axes].child(id).unwrap());
                match by_depth.iter_mut().find(|(d, _)| *d == z) { Some((_, v)) => v.push((axes, id)), None => by_depth.push((z, vec![(axes, id)])) }
            }
            for (z, members) in by_depth { groups.push((z, members, is_bar)); }
        }
        for (zorder, members, is_bar) in groups {
            let stroke = if is_bar {
                let (mut xs, mut lows, mut highs, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), vec![0i64]);
                let (mut colours, mut widths, mut caps, mut clips) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for &(axes, id) in &members {
                    let Some(Child::Segments(s)) = self.fig.axes[axes].child(id) else { return None };
                    if !s.visible { continue; }
                    let m = self.fig.axes[axes].trans_data(self.fig.size_inches, dpi);
                    let mut gx = Vec::new();
                    for &[x, low, high] in &s.segments {
                        let model = &self.fig.axes[axes];
                        let ((sx, s_low), (_, s_high)) = (model.scaled(x, low), model.scaled(x, high));
                        let (x0, y0) = m.apply(sx, s_low);
                        let (x1, y1) = m.apply(sx, s_high);
                        let (ty0, ty1) = (canvas_h - y0, canvas_h - y1);
                        gx.push((x0 + x1) * 0.5);
                        lows.push(ty0.min(ty1));
                        highs.push(ty0.max(ty1));
                    }
                    caps.push(effective_cap_width(&gx, (2.0 * s.capsize * dpi / 72.0).max(0.0)));
                    offsets.push(offsets.last().unwrap() + gx.len() as i64);
                    xs.extend(gx);
                    colours.push(packed(s.facecolor));
                    widths.push((s.linewidth * dpi / 72.0).max(1.0));
                    clips.push(clip_of(self.fig.axes[axes].bbox_exact(self.fig.size_inches, dpi), width, height));
                }
                Stroke { zorder, children: members.clone(), kind: StrokeKind::Bars { xs, lows, highs, offsets, colours, widths, caps, clips } }
            } else {
                self.lines_stroke(zorder, members)?
            };
            strokes.push(stroke);
        }
        strokes.sort_by(|a, b| a.zorder.partial_cmp(&b.zorder).unwrap());
        Some(strokes)
    }

    /// `_native_curve_lines`: the error bars, data lines and fit lines the
    /// kernels may stroke, or None when anything among them needs Agg.
    #[allow(clippy::type_complexity)]
    fn native_lines(&self, prepared: bool) -> Option<(Vec<(usize, ChildId)>, Vec<(usize, ChildId)>, Vec<(usize, ChildId)>)> {
        let mut data: Vec<(usize, ChildId)> = Vec::new();
        for (&axes, records) in &self.curve.series_lines {
            for (line, _, _) in records {
                // A histogram group's bars are no line.
                let visible = matches!(self.fig.axes[axes].child(*line), Some(Child::Line(l)) if l.visible);
                if visible && self.fig.axes[axes].visible { data.push((axes, *line)); }
            }
        }
        let fit: Vec<(usize, ChildId)> = self.fit_lines.iter().copied()
            .filter(|&(axes, id)| self.fig.axes[axes].visible && matches!(self.fig.axes[axes].child(id), Some(Child::Line(l)) if l.visible))
            .collect();
        let mut bars: Vec<(usize, ChildId)> = Vec::new();
        for (&axes, groups) in &self.curve.series_bars {
            for (_, ids) in groups {
                if ids.len() != 1 { return None; }
                if self.fig.axes[axes].visible { bars.push((axes, ids[0])); }
            }
        }
        if matches!(self.spec.semantic(), Surface::Histogram(_)) {
            // A histogram's data is its bars; only a fit over them is stroked.
            if !data.is_empty() || fit.is_empty() { return None; }
        } else if data.is_empty() && !prepared {
            return None;
        }
        for &(axes, line) in data.iter().chain(&fit) {
            let l = self.fig.axes[axes].line(line);
            if !matches!(l.linestyle.as_str(), "-" | "solid") || !matches!(l.marker.as_str(), "None" | "none" | "") { return None; }
            if l.x.iter().chain(&l.y).any(|v| !v.is_finite()) { return None; }
        }
        Some((bars, data, fit))
    }

    /// `_curve_stroke_plan(lines)` as one stroke at `zorder`: each line's
    /// points through its axes' transData (a log axis' non-affine part
    /// first), in canvas rows; None when a point lands nowhere.
    fn lines_stroke(&self, zorder: f64, members: Vec<(usize, ChildId)>) -> Option<Stroke> {
        let (fw, fh) = self.fig.size_px();
        let (width, height) = (fw as usize, fh as usize);
        let dpi = self.fig.dpi;
        let mut raw = Vec::new();
        let mut offsets = vec![0i64];
        let (mut matrices, mut colours, mut widths, mut clips) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &(axes, id) in &members {
            let model = &self.fig.axes[axes];
            let l = model.line(id);
            raw.extend(l.x.iter().zip(&l.y).map(|(&x, &y)| { let (sx, sy) = model.scaled(x, y); [sx, sy] }));
            offsets.push(raw.len() as i64);
            let m = model.trans_data(self.fig.size_inches, dpi);
            matrices.push([m.0[0][0], m.0[0][1], m.0[0][2], m.0[1][0], m.0[1][1], m.0[1][2]]);
            let mut rgba = color::rgba(&l.color);
            if let Some(alpha) = l.alpha { rgba[3] *= alpha; }
            colours.push(packed(rgba));
            widths.push((l.linewidth * dpi / 72.0).max(1.0));
            clips.push(clip_of(model.bbox_exact(self.fig.size_inches, dpi), width, height));
        }
        let mut packed_vertices = vec![[0.0; 2]; raw.len()];
        transform_polylines(&raw, &offsets, &matrices, &vec![1u8; matrices.len()], height as f64, &mut packed_vertices).ok()?;
        if packed_vertices.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) { return None; }
        Some(Stroke { zorder, children: members, kind: StrokeKind::Lines { vertices: packed_vertices, offsets, colours, widths, clips } })
    }

    /// The fit over a histogram's native bars (`_native_curve_lines()`'s fit
    /// lines, one `_curve_stroke_plan`), stroked at the lowest of their depths.
    fn histogram_fit_stroke(&self) -> Option<Stroke> {
        let (_, _, fit) = self.native_lines(false)?;
        let zorder = fit.iter().map(|&(axes, id)| child_zorder(self.fig.axes[axes].child(id).unwrap())).fold(f64::INFINITY, f64::min);
        self.lines_stroke(zorder, fit)
    }

    /// The collections `_raster_prepared_histograms` paints, every painted
    /// surface's in order; None when the command cannot serve the scene (no
    /// command, a surface without one, no bars at all) and the collections draw.
    fn kernel_bars(&self) -> Option<Vec<(usize, ChildId)>> {
        let prepared = self.histogram.prepared.as_ref()?;
        let mut bars = Vec::new();
        for (key, _, _) in self.painted_surfaces() {
            if !prepared.iter().any(|k| *k == key) { return None; }
            bars.extend(self.histogram.of(&key).unwrap_or(&[]).iter().copied()
                .filter(|&(owner, id)| matches!(self.fig.axes[owner].child(id), Some(Child::Bars(_)))));
        }
        (!bars.is_empty()).then_some(bars)
    }

    fn mark_kernel_bars(&mut self, bars: &[(usize, ChildId)], painted: bool) {
        for &(owner, id) in bars {
            if let Some(Child::Bars(b)) = self.fig.axes[owner].child_mut(id) { b.kernel_painted = painted; }
        }
    }

    /// `_raster_prepared_histograms(canvas)`: the bars of every painted
    /// surface (`kernel_bars`) straight into the canvas, as Agg would lay
    /// the collections.
    fn raster_prepared_histograms(&self, backend: &mut AggBackend, painted: &[(usize, ChildId)]) {
        let _span = super::profile::span("compose.kernels");
        let (width, height) = (backend.renderer.width as usize, backend.renderer.height as usize);
        let h = height as f64;
        let (mut edges_px, mut tops_px, mut bases, mut offsets, mut surface_offsets) = (Vec::new(), Vec::new(), Vec::new(), vec![0usize], vec![0usize]);
        let (mut colours, mut clips) = (Vec::new(), Vec::new());
        for (key, axes, _) in self.painted_surfaces() {
            let model = &self.fig.axes[axes];
            let m = model.trans_data(self.fig.size_inches, self.fig.dpi);
            let (a, b, c, d, e, f) = (m.0[0][0], m.0[1][0], m.0[0][1], m.0[1][1], m.0[0][2], m.0[1][2]);
            let affine = !model.xaxis.log && !model.yaxis.log;
            for &(owner, id) in self.histogram.of(&key).unwrap_or(&[]) {
                let Some(Child::Bars(bars)) = self.fig.axes[owner].child(id) else { continue };
                debug_assert!(painted.contains(&(owner, id)));
                colours.push(bars.facecolor().map(|v| (v * 255.0 + 0.5).floor() as u8));
                if affine {
                    edges_px.extend(bars.edges.iter().map(|&x| a * x + e));
                    tops_px.extend(bars.counts.iter().map(|&y| h - (d * y + f)));
                    bases.push(h - f);
                } else {
                    let display = |x: f64, y: f64| { let (sx, sy) = model.scaled(x, y); (a * sx + c * sy + e, b * sx + d * sy + f) };
                    let first = bars.edges.first().copied().unwrap_or(f64::NAN);
                    edges_px.extend(bars.edges.iter().map(|&x| display(x, 0.0).0));
                    tops_px.extend(bars.counts.iter().map(|&y| h - display(first, y).1));
                    bases.push(h - display(first, 0.0).1);
                }
                tops_px.push(f64::NAN);
                offsets.push(offsets.last().unwrap() + bars.edges.len());
            }
            surface_offsets.push(colours.len());
            // Agg rounds the clip box to whole pixels before it clips.
            let bbox = model.bbox_exact(self.fig.size_inches, self.fig.dpi);
            let round = |v: f64| (v + 0.5).floor() as i64;
            clips.push([round(bbox[0]).max(0), round(h - bbox[3]).max(0), round(bbox[2]).min(width as i64), round(h - bbox[1]).min(height as i64)]);
        }
        let mut canvas = Canvas::new(&mut backend.renderer.pixels, width, height).expect("canvas");
        raster_histogram_bars(&mut canvas, &HistogramPlan {
            edges: &edges_px, tops: &tops_px, bases: &bases, offsets: &offsets, surface_offsets: &surface_offsets,
            colours: &colours, clips: &clips,
        }).expect("histogram bars");
    }
}

enum StrokeKind {
    Bars { xs: Vec<f64>, lows: Vec<f64>, highs: Vec<f64>, offsets: Vec<i64>, colours: Vec<[u8; 4]>, widths: Vec<f64>, caps: Vec<f64>, clips: Vec<[i32; 4]> },
    Lines { vertices: Vec<[f64; 2]>, offsets: Vec<i64>, colours: Vec<[u8; 4]>, widths: Vec<f64>, clips: Vec<[i32; 4]> },
}

struct Stroke { zorder: f64, children: Vec<(usize, ChildId)>, kind: StrokeKind }

impl Stroke {
    fn paint(&self, canvas: &mut Canvas<'_>) {
        match &self.kind {
            StrokeKind::Bars { xs, lows, highs, offsets, colours, widths, caps, clips } => {
                if xs.is_empty() { return; }
                let _span = super::profile::span("kernels.stroke.bars");
                raster_error_bars(&ErrorBarBatch { x: xs, y_low: lows, y_high: highs, offsets, colours, widths, cap_widths: caps, clips }, canvas)
                    .expect("error bars");
            }
            StrokeKind::Lines { vertices, offsets, colours, widths, clips } => {
                let _span = super::profile::span("kernels.stroke.lines");
                raster_polylines(&PolylineBatch { vertices, offsets, colours, widths, clips }, canvas).expect("polylines");
            }
        }
    }
}

fn set_artist_visible(artist: &mut Artist, visible: bool) {
    match artist {
        Artist::Text(a) => a.artist.visible = visible,
        Artist::Line(a) => a.artist.visible = visible,
        Artist::Patch(a) => a.artist.visible = visible,
        Artist::Collection(a) => a.artist.visible = visible,
        Artist::QuadMesh(a) => a.artist.visible = visible,
        Artist::Image(a) => a.artist.visible = visible,
        Artist::Group(a) => a.artist.visible = visible,
        Artist::Axes { .. } => {}
    }
}
