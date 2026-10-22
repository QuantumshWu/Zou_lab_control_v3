//! Tick typography for standalone surfaces (`_sync_axes_chrome` and
//! `_fit_tick_label_text`): every drawn axis runs its locator afresh when the
//! chrome's inputs changed, and colliding labels are resolved by a sparser
//! lattice first, the compact size second -- with the label boxes measured
//! by the same text layout the draw uses (zlc-figure's geometry queries).
use zlc_figure::axis::{tick_artists, DrawnTick};
use super::fonts::Math;
use zlc_figure::ir;
use zlc_figure::query::{NullBackend, Scene};
use zlc_figure::transform::Points;

use super::core::Renderer;
use super::mpl::axes::{TickGeometry, TickPolicy, TickSignature};
use crate::ticks::{MemoKey, Placement, MIN_TICK_LABEL_PT};

/// One drawn tick label: which axes and axis, which tick object at which
/// location, which side.
#[derive(Clone, Debug)]
struct Label { axes: usize, x: bool, tick: usize, loc: f64, text: String, side: u8, size: f64 }

fn boxes_meet_inclusive(a: &Points, b: &Points) -> bool { a[2] >= b[0] && a[3] >= b[1] && a[0] <= b[2] && a[1] <= b[3] }

fn outside(b: &Points, figure: &Points) -> bool { b[0] < figure[0] || b[1] < figure[1] || b[2] > figure[2] || b[3] > figure[3] }

impl Renderer {
    pub(crate) fn tick_geometry(&self, axes: usize) -> TickGeometry {
        let a = &self.fig.axes[axes];
        TickGeometry { bbox: a.bbox_exact(self.fig.size_inches, self.fig.dpi), figure: self.fig.size_px(), dpi: self.fig.dpi, room: a.room }
    }

    /// `axis._update_ticks()` on one axis.
    pub(crate) fn update_axis_ticks(&mut self, axes: usize, x: bool) {
        let geometry = self.tick_geometry(axes);
        let measure = self.measure.clone();
        let model = &mut self.fig.axes[axes];
        let view = model.view(x);
        model.axis_mut(x).update_ticks(view, &geometry, measure.as_ref());
    }

    /// What the tick typography is laid out from: the boxes (to a micro-
    /// fraction), the limits, the installed tick policies, the tick
    /// parameters but their label size, and the titles -- as raw numbers.
    fn chrome_signature(&self, axes: &[usize]) -> u64 {
        let mut key = super::hashkey::Key::new();
        key.add(&self.fig.dpi);
        for &index in axes {
            let a = &self.fig.axes[index];
            let b = a.bbox_exact(self.fig.size_inches, self.fig.dpi);
            key.add(&index).add(&b.map(|v| (v * 1e6).round_ties_even() / 1e6)).add(&[a.xlim.0, a.ylim.0, a.xlim.1, a.ylim.1])
                .add(&a.xaxis.signature).add(&a.yaxis.signature);
            for p in [&a.xaxis.major, &a.yaxis.major] {
                let super::mpl::axes::TickParams { size, width, pad, color, labelsize: _, labelsize_kw: _, labelcolor, tickdir, tick1on, tick2on,
                                                   label1on, label2on, grid_on, grid_color, grid_linewidth } = p;
                key.add(&(size, width, pad, color, labelcolor, tickdir, tick1on, tick2on, label1on, label2on, grid_on, grid_color, grid_linewidth));
            }
            key.add(&a.titles[0].text).add(&a.visible);
        }
        key.finish()
    }

    /// `_sync_axes_chrome()`: a grid's overview keeps its own chrome; every
    /// other surface tunes the typography of the axes it draws.
    pub(crate) fn sync_axes_chrome(&mut self) {
        let _span = super::profile::span("present.sync_chrome");
        if self.facet_overview() {
            if !self.facet_cells().is_empty() { self.sync_facet_chrome(); }
            return;
        }
        if self.facet.chrome.is_some() { self.discard_facet_cell_chrome(); }
        let axes: Vec<usize> = (0..self.fig.axes.len()).filter(|&i| self.fig.axes[i].visible && self.fig.axes[i].axison).collect();
        let signature = self.chrome_signature(&axes);
        if self.ticks_layout.as_ref() == Some(&signature) { return; }
        let kw = |p: &super::mpl::axes::TickParams| p.labelsize_kw.then_some(p.labelsize.to_bits());
        let previous: Vec<_> = axes.iter().map(|&i| (kw(&self.fig.axes[i].xaxis.major), kw(&self.fig.axes[i].yaxis.major))).collect();
        for &index in &axes {
            for x in [true, false] {
                self.fig.axes[index].axis_mut(x).forget_tick_memo();
                self.update_axis_ticks(index, x);
            }
        }
        self.fit_tick_label_text(&axes, false);
        self.ticks_layout = Some(signature);
        for (k, &index) in axes.iter().enumerate() {
            let a = &self.fig.axes[index];
            if previous[k] != (kw(&a.xaxis.major), kw(&a.yaxis.major)) { self.chrome_dirty.insert(index); }
        }
    }

    /// The visible, non-empty tick labels of the drawn ticks (`direction_labels`).
    fn direction_labels(&self, axes: &[usize]) -> [Vec<Label>; 2] {
        let mut out: [Vec<Label>; 2] = [Vec::new(), Vec::new()];
        for &index in axes {
            let a = &self.fig.axes[index];
            for (direction, x) in [(0usize, true), (1usize, false)] {
                let axis = a.axis(x);
                for (tick, loc, text) in axis.ticks_in_view(a.view(x)) {
                    if text.is_empty() { continue; }
                    for side in [1u8, 2u8] {
                        let on = if side == 1 { axis.major.label1on } else { axis.major.label2on };
                        let size = axis.label_size(tick, side as usize);
                        if on { out[direction].push(Label { axes: index, x, tick, loc, text: text.clone(), side, size }); }
                    }
                }
            }
        }
        out
    }

    /// The IR the label boxes are measured on: a grid cell's Axis is off, so
    /// its tick template is taken with the axis on (where its labels stand
    /// does not depend on it).
    fn measured_ir(&mut self, axes: &[usize], grid: bool) -> ir::Figure {
        if !grid { return self.fig.to_ir(&self.rc); }
        // Only boxes and texts are measured on a grid: its cells' children
        // and chrome strokes stay out of the IR (a live cell's series alone
        // were most of building it).
        let off: Vec<usize> = axes.iter().copied().filter(|&i| !self.fig.axes[i].axison).collect();
        for &index in &off { self.fig.axes[index].axison = true; }
        let held: Vec<_> = axes.iter().map(|&index| {
            let model = &mut self.fig.axes[index];
            (std::mem::take(&mut model.children), std::mem::take(&mut model.strokes))
        }).collect();
        let figure = self.fig.to_ir(&self.rc);
        for (&index, (children, strokes)) in axes.iter().zip(held) {
            let model = &mut self.fig.axes[index];
            (model.children, model.strokes) = (children, strokes);
        }
        for &index in &off { self.fig.axes[index].axison = false; }
        figure
    }

    /// `_fit_tick_label_text(axes, direction_labels, title_artists)`; `grid`:
    /// a grid's overview, whose equivalent axes keep one lattice and whose
    /// titles are the chrome's.
    pub(crate) fn fit_tick_label_text(&mut self, axes: &[usize], grid: bool) -> bool {
        // Fixed coordinate names restore their authored tier first.
        for &index in axes {
            for x in [true, false] {
                let axis = self.fig.axes[index].axis_mut(x);
                if let Some(TickSignature::Named { size_pt, .. }) = &axis.signature {
                    let size = f64::from_bits(*size_pt);
                    if !axis.major.labelsize_kw || axis.major.labelsize != size { axis.set_label_size_all(size); }
                }
            }
        }
        let mut direction_labels = self.direction_labels(axes);
        if grid {
            // A grid's labels of one direction start at their common size.
            for labels in direction_labels.iter_mut() {
                let size = labels.iter().map(|l| l.size).fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| if s < m { s } else { m })))
                    .unwrap_or(MIN_TICK_LABEL_PT);
                for label in labels.iter_mut() { label.size = size; }
            }
        }
        let mut changed_ticks = false;
        let mut compacted = false;
        let fonts_arc = self.fonts.clone();
        let mut fonts = fonts_arc.lock().unwrap();
        let figure_box: Points = { let (w, h) = self.fig.size_px(); [0.0, 0.0, w, h] };
        // The axes boxes and the titles never move during this pass.
        let ir_figure = self.measured_ir(axes, grid);
        let mut null = NullBackend;
        let mut math = Math;
        let mut scene = Scene::new(&ir_figure, &mut null, &mut *fonts, &mut math).expect("scene");
        let mut obstacles: Vec<Points> = Vec::new();
        for &index in axes { obstacles.push(scene.axes_bbox(index).expect("axes bbox")); }
        if grid {
            for &id in self.facet.chrome.iter().flatten() {
                let Some(position) = self.fig.child_position(id) else { continue };
                if let ir::Artist::Text(title) = &ir_figure.children[position] { obstacles.push(scene.text_extent(title).expect("title extent")); }
            }
        } else {
            for &index in axes {
                let title = &ir_figure.axes[index].titles[0];
                if !title.text.is_empty() { obstacles.push(scene.text_extent(title).expect("title extent")); }
            }
        }
        drop(scene);
        // Tick label boxes as the draw lays them out, one scene per figure state.
        let label_boxes = |fonts: &mut super::fonts::FontSet, ir_figure: &ir::Figure, requests: &[(&Label, &str, f64, f64)]| -> Vec<Points> {
            let mut null = NullBackend;
            let mut math = Math;
            let mut scene = Scene::new(ir_figure, &mut null, fonts, &mut math).expect("scene");
            requests.iter().map(|&(label, text, loc, size)| {
                let axis = if label.x { &ir_figure.axes[label.axes].xaxis } else { &ir_figure.axes[label.axes].yaxis };
                let template = axis.major.as_ref().and_then(|m| m.tick.as_ref()).expect("a drawn axis has a tick template");
                let artists = tick_artists(&DrawnTick { template, loc, label: text, major: true }, label.x);
                let mut t = if label.side == 1 { artists.label1 } else { artists.label2 };
                t.font.size = size;
                scene.text_extent(&t).expect("label extent")
            }).collect()
        };
        loop {
            let labels: Vec<Label> = direction_labels.iter().flatten().cloned().collect();
            if labels.is_empty() { return changed_ticks; }
            let ir_figure = self.measured_ir(axes, grid);
            let requests: Vec<(&Label, &str, f64, f64)> = labels.iter().map(|l| (l, l.text.as_str(), l.loc, l.size)).collect();
            let boxes = label_boxes(&mut fonts, &ir_figure, &requests);
            let n = boxes.len();
            let mut touching = vec![vec![false; n + obstacles.len()]; n];
            for i in 0..n {
                for j in 0..n { touching[i][j] = i != j && boxes_meet_inclusive(&boxes[i], &boxes[j]); }
                for (k, o) in obstacles.iter().enumerate() { touching[i][n + k] = boxes_meet_inclusive(&boxes[i], o); }
            }
            let collisions: Vec<bool> = (0..n).map(|i| touching[i].iter().any(|t| *t) || outside(&boxes[i], &figure_box)).collect();
            if !collisions.iter().any(|c| *c) { break; }
            let mut revised = false;
            if !compacted {
                // A grid's equivalent axes keep one lattice; other surfaces'
                // axes are their own groups. Declared axes with an optional zero first.
                let mut groups: Vec<(String, Vec<(usize, bool)>)> = Vec::new();
                for &index in axes {
                    for x in [true, false] {
                        let a = &self.fig.axes[index];
                        let axis = a.axis(x);
                        let kind = match axis.policy { TickPolicy::Smart(_) => "smart", TickPolicy::Declared { .. } => "declared", _ => continue };
                        let key = if grid {
                            let b = a.bbox_exact(self.fig.size_inches, self.fig.dpi);
                            let size: Vec<u64> = [b[2] - b[0], b[3] - b[1]].iter().map(|v| ((v * 1e6).round_ties_even() / 1e6).to_bits()).collect();
                            let view = a.view(x);
                            format!("{x};{kind};{:?};{:?};{size:?}", [view.0.to_bits(), view.1.to_bits()], axis.signature)
                        } else { format!("{index};{x}") };
                        match groups.iter_mut().find(|(k, _)| *k == key) {
                            Some((_, members)) => members.push((index, x)),
                            None => groups.push((key, vec![(index, x)])),
                        }
                    }
                }
                let zero_optional = |r: &Renderer, g: &(usize, bool)| matches!(&r.fig.axes[g.0].axis(g.1).policy, TickPolicy::Declared { locator } if locator.zero_optional);
                let mut ordered: Vec<Vec<(usize, bool)>> = groups.iter().filter(|g| zero_optional(self, &g.1[0])).map(|g| g.1.clone()).collect();
                ordered.extend(groups.iter().filter(|g| !zero_optional(self, &g.1[0])).map(|g| g.1.clone()));
                for members in ordered {
                    let (index, x) = members[0];
                    let indices: Vec<usize> = (0..n).filter(|&i| members.contains(&(labels[i].axes, labels[i].x))).collect();
                    if indices.is_empty() || !indices.iter().any(|&i| collisions[i]) { continue; }
                    let declared = matches!(self.fig.axes[index].axis(x).policy, TickPolicy::Declared { .. });
                    if declared {
                        let zero_collides = indices.iter().any(|&i| labels[i].loc == 0.0 && collisions[i]);
                        let omit = match &self.fig.axes[index].axis(x).policy {
                            TickPolicy::Declared { locator } => locator.zero_optional && locator.ladder.ticks.contains(&0.0) && zero_collides,
                            _ => unreachable!(),
                        };
                        if omit {
                            for &(member, mx) in &members {
                                if let TickPolicy::Declared { locator } = &mut self.fig.axes[member].axis_mut(mx).policy {
                                    locator.with(|l| l.omit_optional_zero());
                                }
                            }
                            revised = true;
                            break;
                        }
                        continue;
                    }
                    let blockers: Vec<Points> = (0..n).filter(|i| !indices.contains(i)).map(|i| boxes[i])
                        .chain(obstacles.iter().copied()).collect();
                    let mut prototypes: Vec<Label> = Vec::new();
                    for &(member, mx) in &members {
                        for side in [1u8, 2u8] {
                            if let Some(&first) = indices.iter().find(|&&i| labels[i].axes == member && labels[i].x == mx && labels[i].side == side) {
                                prototypes.push(labels[first].clone());
                            }
                        }
                    }
                    let mut best_score: usize = indices.iter().map(|&i| touching[i].iter().filter(|t| **t).count()).sum::<usize>()
                        + indices.iter().filter(|&&i| outside(&boxes[i], &figure_box)).count();
                    let mut best: Option<Placement> = None;
                    let candidates = {
                        let a = &self.fig.axes[index];
                        let axis = a.axis(x);
                        let geometry = axis.geometry(&self.tick_geometry(index));
                        let TickPolicy::Smart(locator) = &axis.policy else { unreachable!() };
                        axis.with_scale(|scale| locator.layout_candidates(a.view(x), geometry, scale, self.measure.as_ref()))
                    };
                    for candidate in candidates {
                        let mut requests: Vec<(&Label, &str, f64, f64)> = Vec::new();
                        for prototype in &prototypes {
                            for (value, text) in candidate.ticks.iter().zip(&candidate.texts) {
                                requests.push((prototype, text.as_str(), *value, prototype.size));
                            }
                        }
                        let trial = label_boxes(&mut fonts, &ir_figure, &requests);
                        let mut score = 0usize;
                        for (i, t) in trial.iter().enumerate() {
                            score += blockers.iter().filter(|b| boxes_meet_inclusive(t, b)).count();
                            score += trial.iter().enumerate().filter(|(j, u)| *j != i && boxes_meet_inclusive(t, u)).count();
                            score += usize::from(outside(t, &figure_box));
                        }
                        if score < best_score {
                            best_score = score;
                            best = Some(candidate);
                            if score == 0 { break; }
                        }
                    }
                    if let Some(best) = best {
                        for &(member, mx) in &members {
                            let geometry = self.tick_geometry(member);
                            let a = &mut self.fig.axes[member];
                            let view = a.view(mx);
                            let axis = a.axis_mut(mx);
                            let key = MemoKey::new(view.0, view.1, axis.geometry(&geometry), axis.scale_serial);
                            let TickPolicy::Smart(locator) = &mut axis.policy else { unreachable!() };
                            let before = locator.ladder.drawn_pt;
                            locator.adopt_layout(&best, view.0 > view.1, Some(key));
                            let after = locator.ladder.drawn_pt;
                            if (after - before).abs() >= 1.0e-9 { axis.set_label_size_all(after); axis.offset_size = after; }
                        }
                        revised = true;
                        break;
                    }
                }
            }
            if revised {
                changed_ticks = true;
                drop(fonts);
                for &index in axes {
                    for x in [true, false] { self.update_axis_ticks(index, x); }
                }
                fonts = fonts_arc.lock().unwrap();
                direction_labels = self.direction_labels(axes);
                continue;
            }
            let mut adjustable = Vec::new();
            for direction in 0..2usize {
                let crowded = labels.iter().zip(&collisions).any(|(l, c)| *c && usize::from(!l.x) == direction);
                if crowded && direction_labels[direction].iter().any(|l| l.size > MIN_TICK_LABEL_PT) { adjustable.push(direction); }
            }
            if adjustable.is_empty() { break; }
            compacted = true;
            for direction in adjustable {
                for label in &mut direction_labels[direction] {
                    label.size = MIN_TICK_LABEL_PT;
                    // `label.set_fontsize`: the tick object keeps it until its
                    // labels are sized again.
                    let axis = self.fig.axes[label.axes].axis_mut(label.x);
                    axis.tick_objects(label.tick + 1)[label.tick].label_size[label.side as usize - 1] = Some(MIN_TICK_LABEL_PT);
                }
            }
        }
        drop(fonts);
        // Commit the accepted typography to the axis templates and locators.
        for (direction, x) in [(0usize, true), (1usize, false)] {
            let size = direction_labels[direction].iter().map(|l| l.size).fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| if s > m { s } else { m })))
                .unwrap_or(MIN_TICK_LABEL_PT);
            for &index in axes {
                let axis = self.fig.axes[index].axis_mut(x);
                match axis.drawn_pt() {
                    Some(drawn) => {
                        let target = if size < drawn { size } else { drawn };
                        if (target - drawn).abs() >= 1.0e-9 {
                            match &mut axis.policy {
                                TickPolicy::Smart(locator) => locator.ladder.drawn_pt = target,
                                TickPolicy::Declared { locator } => locator.ladder.drawn_pt = target,
                                _ => {}
                            }
                            axis.set_label_size_all(target);
                            axis.offset_size = target;
                        }
                    }
                    None => {
                        // `_major_tick_kw.get('labelsize', size)`: an axis whose size
                        // tick_params never set keeps the rc size.
                        if axis.major.labelsize_kw && size < axis.major.labelsize { axis.set_label_size_all(size); }
                    }
                }
            }
        }
        changed_ticks
    }
}
