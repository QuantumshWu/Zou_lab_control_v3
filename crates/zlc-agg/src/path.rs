//! Matplotlib's path sources and `path_converters.h` stages.
//!
//! Each stage keeps exactly the state C++ keeps across `rewind`: a pipeline
//! that is iterated several times (snap detection, fill, hatch, stroke)
//! carries the same leftovers from one pass into the next.
use std::cell::Cell;

use crate::basics::*;
use crate::trans_affine::TransAffine;

/// A borrowed Matplotlib `Path`: N×2 vertices and optional codes.
#[derive(Clone, Copy)]
pub struct PathData<'a> {
    pub vertices: &'a [[f64; 2]],
    pub codes: Option<&'a [u8]>,
    pub should_simplify: bool,
    pub simplify_threshold: f64,
    /// Identity of the vertices array, as `PathIterator::get_id` reports it.
    pub id: usize,
}

impl PathData<'_> {
    pub const EMPTY: PathData<'static> =
        PathData { vertices: &[], codes: None, should_simplify: false, simplify_threshold: 1.0 / 9.0, id: 0 };

    pub fn total_vertices(&self) -> u32 { self.vertices.len() as u32 }
    pub fn has_codes(&self) -> bool { self.codes.is_some() }
}

/// `mpl::PathIterator`.
pub struct PathIterator<'a> { path: PathData<'a>, iterator: u32 }

impl<'a> PathIterator<'a> {
    pub fn new(path: PathData<'a>) -> Self { Self { path, iterator: 0 } }
}

impl VertexSource for PathIterator<'_> {
    fn rewind(&mut self, path_id: u32) { self.iterator = path_id; }

    #[inline]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.iterator >= self.path.total_vertices() {
            *x = 0.0;
            *y = 0.0;
            return PATH_CMD_STOP;
        }
        let idx = self.iterator as usize;
        self.iterator += 1;
        let [vx, vy] = self.path.vertices[idx];
        *x = vx;
        *y = vy;
        match self.path.codes {
            Some(codes) => u32::from(codes[idx]),
            None => if idx == 0 { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO },
        }
    }
}

/// `agg::conv_transform`, reading a transform that may change after construction.
pub struct ConvTransform<'t, S> { source: S, trans: &'t Cell<TransAffine> }

impl<'t, S: VertexSource> ConvTransform<'t, S> {
    pub fn new(source: S, trans: &'t Cell<TransAffine>) -> Self { Self { source, trans } }
}

impl<S: VertexSource> VertexSource for ConvTransform<'_, S> {
    fn rewind(&mut self, path_id: u32) { self.source.rewind(path_id); }

    #[inline]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let cmd = self.source.vertex(x, y);
        if is_vertex(cmd) { self.trans.get().transform(x, y); }
        cmd
    }
}

/// `EmbeddedQueue<N>`.
struct Queue<const N: usize> { read: usize, write: usize, items: [(u32, f64, f64); N] }

impl<const N: usize> Queue<N> {
    fn new() -> Self { Self { read: 0, write: 0, items: [(0, 0.0, 0.0); N] } }
    #[inline(always)]
    fn push(&mut self, cmd: u32, x: f64, y: f64) {
        self.items[self.write] = (cmd, x, y);
        self.write += 1;
    }
    #[inline(always)]
    fn nonempty(&self) -> bool { self.read < self.write }
    #[inline(always)]
    fn pop(&mut self, cmd: &mut u32, x: &mut f64, y: &mut f64) -> bool {
        if self.nonempty() {
            let item = self.items[self.read];
            self.read += 1;
            *cmd = item.0;
            *x = item.1;
            *y = item.2;
            return true;
        }
        self.read = 0;
        self.write = 0;
        false
    }
    #[inline(always)]
    fn clear(&mut self) {
        self.read = 0;
        self.write = 0;
    }
}

const NUM_EXTRA_POINTS: [usize; 16] = [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline(always)]
fn finite(x: f64, y: f64) -> bool { x.is_finite() && y.is_finite() }

/// `PathNanRemover`.
pub struct PathNanRemover<S> {
    source: S,
    queue: Queue<4>,
    remove_nans: bool,
    has_codes: bool,
    valid_segment_exists: bool,
    last_segment_valid: bool,
    was_broken: bool,
    init_x: f64,
    init_y: f64,
}

impl<S: VertexSource> PathNanRemover<S> {
    pub fn new(source: S, remove_nans: bool, has_codes: bool) -> Self {
        Self { source, queue: Queue::new(), remove_nans, has_codes, valid_segment_exists: false,
               last_segment_valid: false, was_broken: false, init_x: f64::NAN, init_y: f64::NAN }
    }
}

impl<S: VertexSource> VertexSource for PathNanRemover<S> {
    fn rewind(&mut self, path_id: u32) {
        self.queue.clear();
        self.source.rewind(path_id);
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut code = 0;
        if !self.remove_nans { return self.source.vertex(x, y); }
        if self.has_codes {
            if self.queue.pop(&mut code, x, y) { return code; }
            let mut needs_move_to = false;
            loop {
                code = self.source.vertex(x, y);
                if code == PATH_CMD_STOP {
                    return code;
                } else if code == CLOSEPOLY && self.valid_segment_exists {
                    if self.was_broken {
                        if self.last_segment_valid && finite(self.init_x, self.init_y) {
                            self.queue.push(PATH_CMD_LINE_TO, self.init_x, self.init_y);
                            break;
                        } else {
                            continue;
                        }
                    } else {
                        return code;
                    }
                } else if code == PATH_CMD_MOVE_TO {
                    self.init_x = *x;
                    self.init_y = *y;
                    self.was_broken = false;
                }
                if needs_move_to { self.queue.push(PATH_CMD_MOVE_TO, *x, *y); }
                let num_extra_points = NUM_EXTRA_POINTS[(code & 0xF) as usize];
                self.last_segment_valid = finite(*x, *y);
                self.queue.push(code, *x, *y);
                for _ in 0..num_extra_points {
                    self.source.vertex(x, y);
                    self.last_segment_valid = self.last_segment_valid && finite(*x, *y);
                    self.queue.push(code, *x, *y);
                }
                if self.last_segment_valid {
                    self.valid_segment_exists = true;
                    break;
                }
                self.was_broken = true;
                self.queue.clear();
                if finite(*x, *y) {
                    self.queue.push(PATH_CMD_MOVE_TO, *x, *y);
                    needs_move_to = false;
                } else {
                    needs_move_to = true;
                }
            }
            if self.queue.pop(&mut code, x, y) { code } else { PATH_CMD_STOP }
        } else {
            code = self.source.vertex(x, y);
            if code == PATH_CMD_STOP || (code == CLOSEPOLY && self.valid_segment_exists) { return code; }
            if !finite(*x, *y) {
                loop {
                    code = self.source.vertex(x, y);
                    if code == PATH_CMD_STOP || (code == CLOSEPOLY && self.valid_segment_exists) { return code; }
                    if finite(*x, *y) { break; }
                }
                return PATH_CMD_MOVE_TO;
            }
            self.valid_segment_exists = true;
            code
        }
    }
}

#[inline(always)]
fn clipping_flags(x: f64, y: f64, b: &RectD) -> u32 {
    u32::from(x > b.x2) | (u32::from(y > b.y2) << 1) | (u32::from(x < b.x1) << 2) | (u32::from(y < b.y1) << 3)
}

#[inline(always)]
fn clip_move_point(x1: f64, y1: f64, x2: f64, y2: f64, b: &RectD, x: &mut f64, y: &mut f64, flags: u32) -> bool {
    if flags & 5 != 0 {
        if x1 == x2 { return false; }
        let bound = if flags & 4 != 0 { b.x1 } else { b.x2 };
        *y = (bound - x1) * (y2 - y1) / (x2 - x1) + y1;
        *x = bound;
    }
    let flags = (u32::from(*y > b.y2) << 1) | (u32::from(*y < b.y1) << 3);
    if flags & 10 != 0 {
        if y1 == y2 { return false; }
        let bound = if flags & 8 != 0 { b.y1 } else { b.y2 };
        *x = (bound - y1) * (x2 - x1) / (y2 - y1) + x1;
        *y = bound;
    }
    true
}

/// `agg::clip_line_segment`.
fn clip_line_segment(x1: &mut f64, y1: &mut f64, x2: &mut f64, y2: &mut f64, b: &RectD) -> u32 {
    let f1 = clipping_flags(*x1, *y1, b);
    let f2 = clipping_flags(*x2, *y2, b);
    let mut ret = 0;
    if f2 | f1 == 0 { return 0; }
    if f1 & 5 != 0 && f1 & 5 == f2 & 5 { return 4; }
    if f1 & 10 != 0 && f1 & 10 == f2 & 10 { return 4; }
    let (tx1, ty1, tx2, ty2) = (*x1, *y1, *x2, *y2);
    if f1 != 0 {
        if !clip_move_point(tx1, ty1, tx2, ty2, b, x1, y1, f1) { return 4; }
        if *x1 == *x2 && *y1 == *y2 { return 4; }
        ret |= 1;
    }
    if f2 != 0 {
        if !clip_move_point(tx1, ty1, tx2, ty2, b, x2, y2, f2) { return 4; }
        if *x1 == *x2 && *y1 == *y2 { return 4; }
        ret |= 2;
    }
    ret
}

/// `PathClipper` against the canvas grown by one pixel on each side.
pub struct PathClipper<S> {
    source: S,
    queue: Queue<3>,
    do_clipping: bool,
    cliprect: RectD,
    last_x: f64,
    last_y: f64,
    moveto: bool,
    init_x: f64,
    init_y: f64,
    has_init: bool,
    was_clipped: bool,
}

impl<S: VertexSource> PathClipper<S> {
    pub fn new(source: S, do_clipping: bool, width: f64, height: f64) -> Self {
        Self { source, queue: Queue::new(), do_clipping, cliprect: RectD::new(-1.0, -1.0, width + 1.0, height + 1.0),
               last_x: f64::NAN, last_y: f64::NAN, moveto: true, init_x: f64::NAN, init_y: f64::NAN,
               has_init: false, was_clipped: false }
    }

    fn draw_clipped_line(&mut self, mut x0: f64, mut y0: f64, mut x1: f64, mut y1: f64, closed: bool) -> bool {
        let moved = clip_line_segment(&mut x0, &mut y0, &mut x1, &mut y1, &self.cliprect);
        self.was_clipped = self.was_clipped || moved != 0;
        if moved < 4 {
            if moved & 1 != 0 || self.moveto { self.queue.push(PATH_CMD_MOVE_TO, x0, y0); }
            self.queue.push(PATH_CMD_LINE_TO, x1, y1);
            if closed && !self.was_clipped { self.queue.push(CLOSEPOLY, x1, y1); }
            self.moveto = false;
            return true;
        }
        false
    }

    #[inline(always)]
    fn last_inside(&self) -> bool {
        self.last_x >= self.cliprect.x1 && self.last_x <= self.cliprect.x2
            && self.last_y >= self.cliprect.y1 && self.last_y <= self.cliprect.y2
    }
}

impl<S: VertexSource> VertexSource for PathClipper<S> {
    fn rewind(&mut self, path_id: u32) {
        self.has_init = false;
        self.was_clipped = false;
        self.moveto = true;
        self.source.rewind(path_id);
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut code = 0;
        if !self.do_clipping { return self.source.vertex(x, y); }
        if self.queue.pop(&mut code, x, y) { return code; }
        'outer: loop {
            code = self.source.vertex(x, y);
            if code == PATH_CMD_STOP { break; }
            match code {
                CLOSEPOLY => {
                    if self.has_init {
                        self.draw_clipped_line(self.last_x, self.last_y, self.init_x, self.init_y, true);
                    } else {
                        self.queue.push(CLOSEPOLY, self.last_x, self.last_y);
                    }
                    if self.queue.nonempty() { break 'outer; }
                }
                PATH_CMD_MOVE_TO => {
                    let mut emit_moveto = false;
                    if self.moveto && self.has_init && self.last_inside() {
                        self.queue.push(PATH_CMD_MOVE_TO, self.last_x, self.last_y);
                        emit_moveto = true;
                    }
                    self.init_x = *x;
                    self.last_x = *x;
                    self.init_y = *y;
                    self.last_y = *y;
                    self.has_init = true;
                    self.moveto = true;
                    self.was_clipped = false;
                    if emit_moveto { break 'outer; }
                }
                PATH_CMD_LINE_TO => {
                    if self.draw_clipped_line(self.last_x, self.last_y, *x, *y, false) {
                        self.last_x = *x;
                        self.last_y = *y;
                        break 'outer;
                    }
                    self.last_x = *x;
                    self.last_y = *y;
                }
                _ => {
                    if self.moveto {
                        self.queue.push(PATH_CMD_MOVE_TO, self.last_x, self.last_y);
                        self.moveto = false;
                    }
                    self.queue.push(code, *x, *y);
                    self.last_x = *x;
                    self.last_y = *y;
                    break 'outer;
                }
            }
        }
        if self.queue.pop(&mut code, x, y) { return code; }
        if self.moveto && self.has_init && self.last_inside() {
            *x = self.last_x;
            *y = self.last_y;
            self.moveto = false;
            return PATH_CMD_MOVE_TO;
        }
        PATH_CMD_STOP
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapMode { Auto, False, True }

/// `PathSnapper`: the construction pass that decides snapping consumes the
/// source once and rewinds it.
pub struct PathSnapper<S> { source: S, snap: bool, snap_value: f64 }

impl<S: VertexSource> PathSnapper<S> {
    pub fn new(mut source: S, snap_mode: SnapMode, total_vertices: u32, stroke_width: f64) -> Self {
        let snap = Self::should_snap(&mut source, snap_mode, total_vertices);
        let mut snap_value = 0.0;
        if snap {
            let is_odd = mpl_round_to_int(stroke_width) % 2;
            snap_value = if is_odd != 0 { 0.5 } else { 0.0 };
        }
        source.rewind(0);
        Self { source, snap, snap_value }
    }

    fn should_snap(path: &mut S, snap_mode: SnapMode, total_vertices: u32) -> bool {
        let (mut x0, mut y0, mut x1, mut y1) = (0.0, 0.0, 0.0, 0.0);
        match snap_mode {
            SnapMode::Auto => {
                if total_vertices > 1024 { return false; }
                let code = path.vertex(&mut x0, &mut y0);
                if code == PATH_CMD_STOP { return false; }
                loop {
                    let code = path.vertex(&mut x1, &mut y1);
                    if code == PATH_CMD_STOP { break; }
                    match code {
                        PATH_CMD_CURVE3 | PATH_CMD_CURVE4 => return false,
                        PATH_CMD_LINE_TO => {
                            if (x0 - x1).abs() >= 1e-4 && (y0 - y1).abs() >= 1e-4 { return false; }
                        }
                        _ => {}
                    }
                    x0 = x1;
                    y0 = y1;
                }
                true
            }
            SnapMode::False => false,
            SnapMode::True => true,
        }
    }

    pub fn is_snapping(&self) -> bool { self.snap }
}

impl<S: VertexSource> VertexSource for PathSnapper<S> {
    fn rewind(&mut self, path_id: u32) { self.source.rewind(path_id); }

    #[inline]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let code = self.source.vertex(x, y);
        if self.snap && is_vertex(code) {
            *x = (*x + 0.5).floor() + self.snap_value;
            *y = (*y + 0.5).floor() + self.snap_value;
        }
        code
    }
}

/// `PathSimplifier`.
pub struct PathSimplifier<S> {
    source: S,
    queue: Queue<9>,
    simplify: bool,
    simplify_threshold: f64,
    moveto: bool,
    after_moveto: bool,
    clipped: bool,
    has_init: bool,
    init_x: f64,
    init_y: f64,
    last_x: f64,
    last_y: f64,
    orig_dx: f64,
    orig_dy: f64,
    orig_d_norm2: f64,
    d_norm2_forward_max: f64,
    d_norm2_backward_max: f64,
    last_forward_max: bool,
    last_backward_max: bool,
    next_x: f64,
    next_y: f64,
    next_backward_x: f64,
    next_backward_y: f64,
    curr_vec_start_x: f64,
    curr_vec_start_y: f64,
}

impl<S: VertexSource> PathSimplifier<S> {
    pub fn new(source: S, do_simplify: bool, simplify_threshold: f64) -> Self {
        Self { source, queue: Queue::new(), simplify: do_simplify,
               simplify_threshold: simplify_threshold * simplify_threshold, moveto: true, after_moveto: false,
               clipped: false, has_init: false, init_x: 0.0, init_y: 0.0, last_x: 0.0, last_y: 0.0, orig_dx: 0.0,
               orig_dy: 0.0, orig_d_norm2: 0.0, d_norm2_forward_max: 0.0, d_norm2_backward_max: 0.0,
               last_forward_max: false, last_backward_max: false, next_x: 0.0, next_y: 0.0, next_backward_x: 0.0,
               next_backward_y: 0.0, curr_vec_start_x: 0.0, curr_vec_start_y: 0.0 }
    }

    fn push(&mut self, x: f64, y: f64) {
        let need_to_push_back = self.d_norm2_backward_max > 0.0;
        if need_to_push_back {
            if self.last_forward_max {
                self.queue.push(PATH_CMD_LINE_TO, self.next_backward_x, self.next_backward_y);
                self.queue.push(PATH_CMD_LINE_TO, self.next_x, self.next_y);
            } else {
                self.queue.push(PATH_CMD_LINE_TO, self.next_x, self.next_y);
                self.queue.push(PATH_CMD_LINE_TO, self.next_backward_x, self.next_backward_y);
            }
        } else {
            self.queue.push(PATH_CMD_LINE_TO, self.next_x, self.next_y);
        }
        if self.clipped {
            self.queue.push(PATH_CMD_MOVE_TO, self.last_x, self.last_y);
        } else if !self.last_forward_max && !self.last_backward_max {
            self.queue.push(PATH_CMD_LINE_TO, self.last_x, self.last_y);
        }
        self.orig_dx = x - self.last_x;
        self.orig_dy = y - self.last_y;
        self.orig_d_norm2 = self.orig_dx * self.orig_dx + self.orig_dy * self.orig_dy;
        self.d_norm2_forward_max = self.orig_d_norm2;
        self.last_forward_max = true;
        let last = self.queue.items[self.queue.write - 1];
        self.curr_vec_start_x = last.1;
        self.curr_vec_start_y = last.2;
        self.last_x = x;
        self.next_x = x;
        self.last_y = y;
        self.next_y = y;
        self.d_norm2_backward_max = 0.0;
        self.last_backward_max = false;
        self.clipped = false;
    }
}

impl<S: VertexSource> VertexSource for PathSimplifier<S> {
    fn rewind(&mut self, path_id: u32) {
        self.queue.clear();
        self.moveto = true;
        self.source.rewind(path_id);
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut cmd = 0;
        if !self.simplify { return self.source.vertex(x, y); }
        if self.queue.pop(&mut cmd, x, y) { return cmd; }
        loop {
            cmd = self.source.vertex(x, y);
            if cmd == PATH_CMD_STOP { break; }
            if self.moveto || cmd == PATH_CMD_MOVE_TO {
                if self.orig_d_norm2 != 0.0 && !self.after_moveto { self.push(*x, *y); }
                self.after_moveto = true;
                if finite(*x, *y) {
                    self.has_init = true;
                    self.init_x = *x;
                    self.init_y = *y;
                } else {
                    self.has_init = false;
                }
                self.last_x = *x;
                self.last_y = *y;
                self.moveto = false;
                self.orig_d_norm2 = 0.0;
                self.d_norm2_backward_max = 0.0;
                self.clipped = true;
                if self.queue.nonempty() { break; }
                continue;
            }
            self.after_moveto = false;
            if is_close(cmd) {
                if self.has_init {
                    *x = self.init_x;
                    *y = self.init_y;
                } else {
                    continue;
                }
            }
            if self.orig_d_norm2 == 0.0 {
                if self.clipped {
                    self.queue.push(PATH_CMD_MOVE_TO, self.last_x, self.last_y);
                    self.clipped = false;
                }
                self.orig_dx = *x - self.last_x;
                self.orig_dy = *y - self.last_y;
                self.orig_d_norm2 = self.orig_dx * self.orig_dx + self.orig_dy * self.orig_dy;
                self.d_norm2_forward_max = self.orig_d_norm2;
                self.d_norm2_backward_max = 0.0;
                self.last_forward_max = true;
                self.last_backward_max = false;
                self.curr_vec_start_x = self.last_x;
                self.curr_vec_start_y = self.last_y;
                self.last_x = *x;
                self.next_x = *x;
                self.last_y = *y;
                self.next_y = *y;
                continue;
            }
            let totdx = *x - self.curr_vec_start_x;
            let totdy = *y - self.curr_vec_start_y;
            let totdot = self.orig_dx * totdx + self.orig_dy * totdy;
            let paradx = totdot * self.orig_dx / self.orig_d_norm2;
            let parady = totdot * self.orig_dy / self.orig_d_norm2;
            let perpdx = totdx - paradx;
            let perpdy = totdy - parady;
            let perpd_norm2 = perpdx * perpdx + perpdy * perpdy;
            if perpd_norm2 < self.simplify_threshold {
                let parad_norm2 = paradx * paradx + parady * parady;
                self.last_forward_max = false;
                self.last_backward_max = false;
                if totdot > 0.0 {
                    if parad_norm2 > self.d_norm2_forward_max {
                        self.last_forward_max = true;
                        self.d_norm2_forward_max = parad_norm2;
                        self.next_x = *x;
                        self.next_y = *y;
                    }
                } else if parad_norm2 > self.d_norm2_backward_max {
                    self.last_backward_max = true;
                    self.d_norm2_backward_max = parad_norm2;
                    self.next_backward_x = *x;
                    self.next_backward_y = *y;
                }
                self.last_x = *x;
                self.last_y = *y;
                continue;
            }
            self.push(*x, *y);
            break;
        }
        if cmd == PATH_CMD_STOP {
            if self.orig_d_norm2 != 0.0 {
                let c = if self.moveto || self.after_moveto { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO };
                self.queue.push(c, self.next_x, self.next_y);
                if self.d_norm2_backward_max > 0.0 {
                    let c = if self.moveto || self.after_moveto { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO };
                    self.queue.push(c, self.next_backward_x, self.next_backward_y);
                }
                self.moveto = false;
            }
            let c = if self.moveto || self.after_moveto { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO };
            self.queue.push(c, self.last_x, self.last_y);
            self.moveto = false;
            self.queue.push(PATH_CMD_STOP, 0.0, 0.0);
        }
        if self.queue.pop(&mut cmd, x, y) { cmd } else { PATH_CMD_STOP }
    }
}

/// `Sketch` with its `conv_segmentator` over the same source.
pub struct Sketch<S> {
    source: S,
    scale: f64,
    // conv_adaptor_vpgen<vpgen_segmentator>
    seg_start_x: f64,
    seg_start_y: f64,
    seg_poly_flags: u32,
    seg_vertices: i32,
    // vpgen_segmentator
    vp_x1: f64,
    vp_y1: f64,
    vp_dx: f64,
    vp_dy: f64,
    vp_dl: f64,
    vp_ddl: f64,
    vp_cmd: u32,
    last_x: f64,
    last_y: f64,
    has_last: bool,
    p: f64,
    rand: u32,
    p_scale: f64,
    log_randomness: f64,
}

impl<S: VertexSource> Sketch<S> {
    pub fn new(source: S, scale: f64, length: f64, randomness: f64) -> Self {
        let mut sketch = Self { source, scale, seg_start_x: 0.0, seg_start_y: 0.0, seg_poly_flags: 0, seg_vertices: 0,
                                vp_x1: 0.0, vp_y1: 0.0, vp_dx: 0.0, vp_dy: 0.0, vp_dl: 0.0, vp_ddl: 0.0,
                                vp_cmd: PATH_CMD_STOP, last_x: 0.0, last_y: 0.0, has_last: false, p: 0.0, rand: 0,
                                p_scale: 0.0, log_randomness: 0.0 };
        sketch.rewind(0);
        let d_m_pi = 3.14159265358979323846;
        sketch.p_scale = if length <= f64::EPSILON || randomness <= f64::EPSILON { 0.0 }
                         else { (2.0 * d_m_pi) / (length * randomness) };
        sketch.log_randomness = if randomness <= f64::EPSILON { 0.0 } else { 2.0 * randomness.ln() };
        sketch
    }

    fn vp_move_to(&mut self, x: f64, y: f64) {
        self.vp_x1 = x;
        self.vp_y1 = y;
        self.vp_dx = 0.0;
        self.vp_dy = 0.0;
        self.vp_dl = 2.0;
        self.vp_ddl = 2.0;
        self.vp_cmd = PATH_CMD_MOVE_TO;
    }

    fn vp_line_to(&mut self, x: f64, y: f64) {
        self.vp_x1 += self.vp_dx;
        self.vp_y1 += self.vp_dy;
        self.vp_dx = x - self.vp_x1;
        self.vp_dy = y - self.vp_y1;
        let mut len = (self.vp_dx * self.vp_dx + self.vp_dy * self.vp_dy).sqrt() * 1.0;
        if len < 1e-30 { len = 1e-30; }
        self.vp_ddl = 1.0 / len;
        self.vp_dl = if self.vp_cmd == PATH_CMD_MOVE_TO { 0.0 } else { self.vp_ddl };
        if self.vp_cmd == PATH_CMD_STOP { self.vp_cmd = PATH_CMD_LINE_TO; }
    }

    fn vp_vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.vp_cmd == PATH_CMD_STOP { return PATH_CMD_STOP; }
        let cmd = self.vp_cmd;
        self.vp_cmd = PATH_CMD_LINE_TO;
        if self.vp_dl >= 1.0 - self.vp_ddl {
            self.vp_dl = 1.0;
            self.vp_cmd = PATH_CMD_STOP;
            *x = self.vp_x1 + self.vp_dx;
            *y = self.vp_y1 + self.vp_dy;
            return cmd;
        }
        *x = self.vp_x1 + self.vp_dx * self.vp_dl;
        *y = self.vp_y1 + self.vp_dy * self.vp_dl;
        self.vp_dl += self.vp_ddl;
        cmd
    }

    /// `conv_adaptor_vpgen<VertexSource, vpgen_segmentator>::vertex`.
    fn segmented_vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let mut cmd;
        loop {
            cmd = self.vp_vertex(x, y);
            if !is_stop(cmd) { break; }
            if self.seg_poly_flags != 0 {
                *x = 0.0;
                *y = 0.0;
                cmd = self.seg_poly_flags;
                self.seg_poly_flags = 0;
                break;
            }
            if self.seg_vertices < 0 {
                if self.seg_vertices < -1 {
                    self.seg_vertices = 0;
                    return PATH_CMD_STOP;
                }
                self.vp_move_to(self.seg_start_x, self.seg_start_y);
                self.seg_vertices = 1;
                continue;
            }
            let (mut tx, mut ty) = (0.0, 0.0);
            cmd = self.source.vertex(&mut tx, &mut ty);
            if is_vertex(cmd) {
                if is_move_to(cmd) {
                    self.vp_move_to(tx, ty);
                    self.seg_start_x = tx;
                    self.seg_start_y = ty;
                    self.seg_vertices = 1;
                } else {
                    self.vp_line_to(tx, ty);
                    self.seg_vertices += 1;
                }
            } else if is_end_poly(cmd) {
                self.seg_poly_flags = cmd;
                if is_closed(cmd) {
                    if self.seg_vertices > 2 { self.vp_line_to(self.seg_start_x, self.seg_start_y); }
                    self.seg_vertices = 0;
                }
            } else {
                break;
            }
        }
        cmd
    }

    fn next_double(&mut self) -> f64 {
        self.rand = self.rand.wrapping_mul(214013).wrapping_add(2531011);
        f64::from(self.rand) / 4294967296.0
    }
}

impl<S: VertexSource> VertexSource for Sketch<S> {
    fn rewind(&mut self, path_id: u32) {
        self.has_last = false;
        self.p = 0.0;
        if self.scale != 0.0 {
            self.rand = 0;
            self.source.rewind(path_id);
            self.vp_cmd = PATH_CMD_STOP;
            self.seg_start_x = 0.0;
            self.seg_start_y = 0.0;
            self.seg_poly_flags = 0;
            self.seg_vertices = 0;
        } else {
            self.source.rewind(path_id);
        }
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.scale == 0.0 { return self.source.vertex(x, y); }
        let code = self.segmented_vertex(x, y);
        if code == PATH_CMD_MOVE_TO {
            self.has_last = false;
            self.p = 0.0;
        }
        if self.has_last {
            let d_rand = self.next_double();
            self.p += (d_rand * self.log_randomness).exp();
            let den = self.last_x - *x;
            let num = self.last_y - *y;
            let mut len = num * num + den * den;
            self.last_x = *x;
            self.last_y = *y;
            if len != 0.0 {
                len = len.sqrt();
                let r = (self.p * self.p_scale).sin() * self.scale;
                let roverlen = r / len;
                *x += roverlen * num;
                *y -= roverlen * den;
            }
        } else {
            self.last_x = *x;
            self.last_y = *y;
        }
        self.has_last = true;
        code
    }
}

/// One quadrilateral of a mesh: (m, n), (m, n+1), (m+1, n+1), (m+1, n), (m, n).
pub struct QuadMeshPathIterator<'a> { iterator: u32, m: usize, n: usize, coordinates: &'a [[f64; 2]], columns: usize }

impl<'a> QuadMeshPathIterator<'a> {
    /// `coordinates` is the (rows × columns) mesh in row order.
    pub fn new(m: usize, n: usize, coordinates: &'a [[f64; 2]], columns: usize) -> Self {
        Self { iterator: 0, m, n, coordinates, columns }
    }
}

impl VertexSource for QuadMeshPathIterator<'_> {
    fn rewind(&mut self, path_id: u32) { self.iterator = path_id; }

    #[inline]
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.iterator >= 5 { return PATH_CMD_STOP; }
        let idx = self.iterator as usize;
        self.iterator += 1;
        let m = self.m + ((idx & 0x2) >> 1);
        let n = self.n + (((idx + 1) & 0x2) >> 1);
        let [vx, vy] = self.coordinates[n * self.columns + m];
        *x = vx;
        *y = vy;
        if idx != 0 { PATH_CMD_LINE_TO } else { PATH_CMD_MOVE_TO }
    }
}

/// The unclosed rectangle `move_to(0,0)` … `line_to(0,0)` used for images.
pub struct RectPath { w: f64, h: f64, iterator: usize }

impl RectPath {
    pub fn new(w: f64, h: f64) -> Self { Self { w, h, iterator: 0 } }
}

impl VertexSource for RectPath {
    fn rewind(&mut self, _path_id: u32) { self.iterator = 0; }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let points = [(0.0, 0.0), (self.w, 0.0), (self.w, self.h), (0.0, self.h), (0.0, 0.0)];
        if self.iterator >= points.len() { return PATH_CMD_STOP; }
        (*x, *y) = points[self.iterator];
        self.iterator += 1;
        if self.iterator == 1 { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO }
    }
}

/// `conv_transform` with a fixed transform, for internal shapes.
pub struct Transformed<S> { pub source: S, pub trans: TransAffine }

impl<S: VertexSource> VertexSource for Transformed<S> {
    fn rewind(&mut self, path_id: u32) { self.source.rewind(path_id); }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        let cmd = self.source.vertex(x, y);
        if is_vertex(cmd) { self.trans.transform(x, y); }
        cmd
    }
}
