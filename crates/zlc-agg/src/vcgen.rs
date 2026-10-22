//! `conv_stroke`/`conv_dash`: `conv_adaptor_vcgen` with `vcgen_stroke`
//! (`math_stroke` caps, joins and miter limits) or `vcgen_dash`.
//! Matplotlib never shortens paths, so the shortening step is absent.
use crate::basics::*;

#[derive(Clone, Copy, Debug)]
pub struct VertexDist { pub x: f64, pub y: f64, pub dist: f64 }

impl VertexDist {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y, dist: 0.0 } }

    /// `vertex_dist::operator()`: records the distance to `val`.
    #[inline(always)]
    fn test(&mut self, val: &VertexDist) -> bool {
        self.dist = calc_distance(self.x, self.y, val.x, val.y);
        let ret = self.dist > VERTEX_DIST_EPSILON;
        if !ret { self.dist = 1.0 / VERTEX_DIST_EPSILON; }
        ret
    }
}

/// `vertex_sequence<vertex_dist>`: drops coinciding vertices as they arrive.
#[derive(Default)]
pub struct VertexSequence { v: Vec<VertexDist> }

impl VertexSequence {
    #[inline]
    fn test_pair(&mut self, a: usize, b: usize) -> bool {
        let other = self.v[b];
        self.v[a].test(&other)
    }
    pub fn len(&self) -> usize { self.v.len() }
    pub fn is_empty(&self) -> bool { self.v.is_empty() }
    pub fn remove_all(&mut self) { self.v.clear(); }
    fn remove_last(&mut self) { self.v.pop(); }

    pub fn add(&mut self, val: VertexDist) {
        let n = self.v.len();
        if n > 1 && !self.test_pair(n - 2, n - 1) { self.remove_last(); }
        self.v.push(val);
    }

    pub fn modify_last(&mut self, val: VertexDist) {
        self.remove_last();
        self.add(val);
    }

    pub fn close(&mut self, closed: bool) {
        while self.v.len() > 1 {
            let n = self.v.len();
            if self.test_pair(n - 2, n - 1) { break; }
            let t = self.v[n - 1];
            self.remove_last();
            self.modify_last(t);
        }
        if closed {
            while self.v.len() > 1 {
                let n = self.v.len();
                if self.test_pair(n - 1, 0) { break; }
                self.remove_last();
            }
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> VertexDist { self.v[i] }
    #[inline(always)]
    fn prev(&self, i: usize) -> VertexDist { self.v[(i + self.v.len() - 1) % self.v.len()] }
    #[inline(always)]
    fn next(&self, i: usize) -> VertexDist { self.v[(i + 1) % self.v.len()] }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineCap { Butt, Square, Round }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineJoin { Miter, MiterRevert, Round, Bevel, MiterRound }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerJoin { Bevel, Miter, Jag, Round }

pub struct MathStroke {
    width: f64,
    width_abs: f64,
    width_eps: f64,
    width_sign: i32,
    miter_limit: f64,
    inner_miter_limit: f64,
    approx_scale: f64,
    line_cap: LineCap,
    line_join: LineJoin,
    inner_join: InnerJoin,
}

impl Default for MathStroke {
    fn default() -> Self {
        Self { width: 0.5, width_abs: 0.5, width_eps: 0.5 / 1024.0, width_sign: 1, miter_limit: 4.0,
               inner_miter_limit: 1.01, approx_scale: 1.0, line_cap: LineCap::Butt, line_join: LineJoin::Miter,
               inner_join: InnerJoin::Miter }
    }
}

impl MathStroke {
    pub fn set_width(&mut self, w: f64) {
        self.width = w * 0.5;
        if self.width < 0.0 {
            self.width_abs = -self.width;
            self.width_sign = -1;
        } else {
            self.width_abs = self.width;
            self.width_sign = 1;
        }
        self.width_eps = self.width / 1024.0;
    }

    fn calc_arc(&self, vc: &mut Vec<PointD>, x: f64, y: f64, dx1: f64, dy1: f64, dx2: f64, dy2: f64) {
        let sign = f64::from(self.width_sign);
        let mut a1 = (dy1 * sign).atan2(dx1 * sign);
        let mut a2 = (dy2 * sign).atan2(dx2 * sign);
        let mut da = (self.width_abs / (self.width_abs + 0.125 / self.approx_scale)).acos() * 2.0;
        vc.push(PointD::new(x + dx1, y + dy1));
        if self.width_sign > 0 {
            if a1 > a2 { a2 += 2.0 * PI; }
            let n = trunc_i32((a2 - a1) / da);
            da = (a2 - a1) / f64::from(n + 1);
            a1 += da;
            for _ in 0..n {
                vc.push(PointD::new(x + a1.cos() * self.width, y + a1.sin() * self.width));
                a1 += da;
            }
        } else {
            if a1 < a2 { a2 -= 2.0 * PI; }
            let n = trunc_i32((a1 - a2) / da);
            da = (a1 - a2) / f64::from(n + 1);
            a1 -= da;
            for _ in 0..n {
                vc.push(PointD::new(x + a1.cos() * self.width, y + a1.sin() * self.width));
                a1 -= da;
            }
        }
        vc.push(PointD::new(x + dx2, y + dy2));
    }

    #[allow(clippy::too_many_arguments)]
    fn calc_miter(&self, vc: &mut Vec<PointD>, v0: &VertexDist, v1: &VertexDist, v2: &VertexDist,
                  dx1: f64, dy1: f64, dx2: f64, dy2: f64, lj: LineJoin, mut mlimit: f64, dbevel: f64) {
        let mut xi = v1.x;
        let mut yi = v1.y;
        let mut di = 1.0;
        let lim = self.width_abs * mlimit;
        let mut miter_limit_exceeded = true;
        let mut intersection_failed = true;
        if calc_intersection(v0.x + dx1, v0.y - dy1, v1.x + dx1, v1.y - dy1,
                             v1.x + dx2, v1.y - dy2, v2.x + dx2, v2.y - dy2, &mut xi, &mut yi) {
            di = calc_distance(v1.x, v1.y, xi, yi);
            if di <= lim {
                vc.push(PointD::new(xi, yi));
                miter_limit_exceeded = false;
            }
            intersection_failed = false;
        } else {
            let x2 = v1.x + dx1;
            let y2 = v1.y - dy1;
            if (cross_product(v0.x, v0.y, v1.x, v1.y, x2, y2) < 0.0)
                == (cross_product(v1.x, v1.y, v2.x, v2.y, x2, y2) < 0.0) {
                vc.push(PointD::new(v1.x + dx1, v1.y - dy1));
                miter_limit_exceeded = false;
            }
        }
        if miter_limit_exceeded {
            match lj {
                LineJoin::MiterRevert => {
                    vc.push(PointD::new(v1.x + dx1, v1.y - dy1));
                    vc.push(PointD::new(v1.x + dx2, v1.y - dy2));
                }
                LineJoin::MiterRound => self.calc_arc(vc, v1.x, v1.y, dx1, -dy1, dx2, -dy2),
                _ => {
                    if intersection_failed {
                        mlimit *= f64::from(self.width_sign);
                        vc.push(PointD::new(v1.x + dx1 + dy1 * mlimit, v1.y - dy1 + dx1 * mlimit));
                        vc.push(PointD::new(v1.x + dx2 - dy2 * mlimit, v1.y - dy2 - dx2 * mlimit));
                    } else {
                        let x1 = v1.x + dx1;
                        let y1 = v1.y - dy1;
                        let x2 = v1.x + dx2;
                        let y2 = v1.y - dy2;
                        di = (lim - dbevel) / (di - dbevel);
                        vc.push(PointD::new(x1 + (xi - x1) * di, y1 + (yi - y1) * di));
                        vc.push(PointD::new(x2 + (xi - x2) * di, y2 + (yi - y2) * di));
                    }
                }
            }
        }
    }

    fn calc_cap(&self, vc: &mut Vec<PointD>, v0: &VertexDist, v1: &VertexDist, len: f64) {
        vc.clear();
        let mut dx1 = (v1.y - v0.y) / len;
        let mut dy1 = (v1.x - v0.x) / len;
        let mut dx2 = 0.0;
        let mut dy2 = 0.0;
        dx1 *= self.width;
        dy1 *= self.width;
        if self.line_cap != LineCap::Round {
            if self.line_cap == LineCap::Square {
                dx2 = dy1 * f64::from(self.width_sign);
                dy2 = dx1 * f64::from(self.width_sign);
            }
            vc.push(PointD::new(v0.x - dx1 - dx2, v0.y + dy1 - dy2));
            vc.push(PointD::new(v0.x + dx1 - dx2, v0.y - dy1 - dy2));
        } else {
            let mut da = (self.width_abs / (self.width_abs + 0.125 / self.approx_scale)).acos() * 2.0;
            let n = trunc_i32(PI / da);
            da = PI / f64::from(n + 1);
            vc.push(PointD::new(v0.x - dx1, v0.y + dy1));
            if self.width_sign > 0 {
                let mut a1 = dy1.atan2(-dx1);
                a1 += da;
                for _ in 0..n {
                    vc.push(PointD::new(v0.x + a1.cos() * self.width, v0.y + a1.sin() * self.width));
                    a1 += da;
                }
            } else {
                let mut a1 = (-dy1).atan2(dx1);
                a1 -= da;
                for _ in 0..n {
                    vc.push(PointD::new(v0.x + a1.cos() * self.width, v0.y + a1.sin() * self.width));
                    a1 -= da;
                }
            }
            vc.push(PointD::new(v0.x + dx1, v0.y - dy1));
        }
    }

    fn calc_join(&self, vc: &mut Vec<PointD>, v0: &VertexDist, v1: &VertexDist, v2: &VertexDist, len1: f64, len2: f64) {
        let dx1 = self.width * (v1.y - v0.y) / len1;
        let dy1 = self.width * (v1.x - v0.x) / len1;
        let dx2 = self.width * (v2.y - v1.y) / len2;
        let dy2 = self.width * (v2.x - v1.x) / len2;
        vc.clear();
        let cp = cross_product(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y);
        if (cp > VERTEX_DIST_EPSILON && self.width > 0.0) || (cp < -VERTEX_DIST_EPSILON && self.width < 0.0) {
            let mut limit = (if len1 < len2 { len1 } else { len2 }) / self.width_abs;
            if limit < self.inner_miter_limit { limit = self.inner_miter_limit; }
            match self.inner_join {
                InnerJoin::Bevel => {
                    vc.push(PointD::new(v1.x + dx1, v1.y - dy1));
                    vc.push(PointD::new(v1.x + dx2, v1.y - dy2));
                }
                InnerJoin::Miter => self.calc_miter(vc, v0, v1, v2, dx1, dy1, dx2, dy2, LineJoin::MiterRevert, limit, 0.0),
                InnerJoin::Jag | InnerJoin::Round => {
                    let cp = (dx1 - dx2) * (dx1 - dx2) + (dy1 - dy2) * (dy1 - dy2);
                    if cp < len1 * len1 && cp < len2 * len2 {
                        self.calc_miter(vc, v0, v1, v2, dx1, dy1, dx2, dy2, LineJoin::MiterRevert, limit, 0.0);
                    } else if self.inner_join == InnerJoin::Jag {
                        vc.push(PointD::new(v1.x + dx1, v1.y - dy1));
                        vc.push(PointD::new(v1.x, v1.y));
                        vc.push(PointD::new(v1.x + dx2, v1.y - dy2));
                    } else {
                        vc.push(PointD::new(v1.x + dx1, v1.y - dy1));
                        vc.push(PointD::new(v1.x, v1.y));
                        self.calc_arc(vc, v1.x, v1.y, dx2, -dy2, dx1, -dy1);
                        vc.push(PointD::new(v1.x, v1.y));
                        vc.push(PointD::new(v1.x + dx2, v1.y - dy2));
                    }
                }
            }
        } else {
            let mut dx = (dx1 + dx2) / 2.0;
            let mut dy = (dy1 + dy2) / 2.0;
            let dbevel = (dx * dx + dy * dy).sqrt();
            if (self.line_join == LineJoin::Round || self.line_join == LineJoin::Bevel)
                && self.approx_scale * (self.width_abs - dbevel) < self.width_eps {
                if calc_intersection(v0.x + dx1, v0.y - dy1, v1.x + dx1, v1.y - dy1,
                                     v1.x + dx2, v1.y - dy2, v2.x + dx2, v2.y - dy2, &mut dx, &mut dy) {
                    vc.push(PointD::new(dx, dy));
                } else {
                    vc.push(PointD::new(v1.x + dx1, v1.y - dy1));
                }
                return;
            }
            match self.line_join {
                LineJoin::Miter | LineJoin::MiterRevert | LineJoin::MiterRound =>
                    self.calc_miter(vc, v0, v1, v2, dx1, dy1, dx2, dy2, self.line_join, self.miter_limit, dbevel),
                LineJoin::Round => self.calc_arc(vc, v1.x, v1.y, dx1, -dy1, dx2, -dy2),
                LineJoin::Bevel => {
                    vc.push(PointD::new(v1.x + dx1, v1.y - dy1));
                    vc.push(PointD::new(v1.x + dx2, v1.y - dy2));
                }
            }
        }
    }
}

/// The vertex-generator half of `conv_adaptor_vcgen`.
pub trait Generator {
    fn remove_all(&mut self);
    fn add_vertex(&mut self, x: f64, y: f64, cmd: u32);
    fn rewind(&mut self, path_id: u32);
    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StrokeStatus { Initial, Ready, Cap1, Cap2, Outline1, CloseFirst, Outline2, OutVertices, EndPoly1, EndPoly2, Stop }

pub struct VcgenStroke {
    pub stroker: MathStroke,
    src_vertices: VertexSequence,
    out_vertices: Vec<PointD>,
    closed: u32,
    status: StrokeStatus,
    prev_status: StrokeStatus,
    src_vertex: usize,
    out_vertex: usize,
}

impl Default for VcgenStroke {
    fn default() -> Self {
        Self { stroker: MathStroke::default(), src_vertices: VertexSequence::default(), out_vertices: Vec::new(),
               closed: 0, status: StrokeStatus::Initial, prev_status: StrokeStatus::Initial, src_vertex: 0, out_vertex: 0 }
    }
}

impl Generator for VcgenStroke {
    fn remove_all(&mut self) {
        self.src_vertices.remove_all();
        self.closed = 0;
        self.status = StrokeStatus::Initial;
    }

    fn add_vertex(&mut self, x: f64, y: f64, cmd: u32) {
        self.status = StrokeStatus::Initial;
        if is_move_to(cmd) {
            self.src_vertices.modify_last(VertexDist::new(x, y));
        } else if is_vertex(cmd) {
            self.src_vertices.add(VertexDist::new(x, y));
        } else {
            self.closed = get_close_flag(cmd);
        }
    }

    fn rewind(&mut self, _path_id: u32) {
        if self.status == StrokeStatus::Initial {
            self.src_vertices.close(self.closed != 0);
            if self.src_vertices.len() < 3 { self.closed = 0; }
        }
        self.status = StrokeStatus::Ready;
        self.src_vertex = 0;
        self.out_vertex = 0;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        use StrokeStatus as S;
        let mut cmd = PATH_CMD_LINE_TO;
        while !is_stop(cmd) {
            match self.status {
                S::Initial | S::Ready => {
                    if self.status == S::Initial { self.rewind(0); }
                    if self.src_vertices.len() < 2 + usize::from(self.closed != 0) {
                        cmd = PATH_CMD_STOP;
                        continue;
                    }
                    self.status = if self.closed != 0 { S::Outline1 } else { S::Cap1 };
                    cmd = PATH_CMD_MOVE_TO;
                    self.src_vertex = 0;
                    self.out_vertex = 0;
                }
                S::Cap1 => {
                    let (v0, v1) = (self.src_vertices.get(0), self.src_vertices.get(1));
                    self.stroker.calc_cap(&mut self.out_vertices, &v0, &v1, v0.dist);
                    self.src_vertex = 1;
                    self.prev_status = S::Outline1;
                    self.status = S::OutVertices;
                    self.out_vertex = 0;
                }
                S::Cap2 => {
                    let n = self.src_vertices.len();
                    let (v0, v1) = (self.src_vertices.get(n - 1), self.src_vertices.get(n - 2));
                    self.stroker.calc_cap(&mut self.out_vertices, &v0, &v1, v1.dist);
                    self.prev_status = S::Outline2;
                    self.status = S::OutVertices;
                    self.out_vertex = 0;
                }
                S::Outline1 => {
                    if self.closed != 0 {
                        if self.src_vertex >= self.src_vertices.len() {
                            self.prev_status = S::CloseFirst;
                            self.status = S::EndPoly1;
                            continue;
                        }
                    } else if self.src_vertex >= self.src_vertices.len() - 1 {
                        self.status = S::Cap2;
                        continue;
                    }
                    let i = self.src_vertex;
                    let (v0, v1, v2) = (self.src_vertices.prev(i), self.src_vertices.get(i), self.src_vertices.next(i));
                    self.stroker.calc_join(&mut self.out_vertices, &v0, &v1, &v2, v0.dist, v1.dist);
                    self.src_vertex += 1;
                    self.prev_status = self.status;
                    self.status = S::OutVertices;
                    self.out_vertex = 0;
                }
                S::CloseFirst | S::Outline2 => {
                    if self.status == S::CloseFirst {
                        self.status = S::Outline2;
                        cmd = PATH_CMD_MOVE_TO;
                    }
                    if self.src_vertex <= usize::from(self.closed == 0) {
                        self.status = S::EndPoly2;
                        self.prev_status = S::Stop;
                        continue;
                    }
                    self.src_vertex -= 1;
                    let i = self.src_vertex;
                    let (v0, v1, v2) = (self.src_vertices.next(i), self.src_vertices.get(i), self.src_vertices.prev(i));
                    self.stroker.calc_join(&mut self.out_vertices, &v0, &v1, &v2, v1.dist, v2.dist);
                    self.prev_status = self.status;
                    self.status = S::OutVertices;
                    self.out_vertex = 0;
                }
                S::OutVertices => {
                    if self.out_vertex >= self.out_vertices.len() {
                        self.status = self.prev_status;
                    } else {
                        let c = self.out_vertices[self.out_vertex];
                        self.out_vertex += 1;
                        *x = c.x;
                        *y = c.y;
                        return cmd;
                    }
                }
                S::EndPoly1 => {
                    self.status = self.prev_status;
                    return PATH_CMD_END_POLY | PATH_FLAGS_CLOSE | PATH_FLAGS_CCW;
                }
                S::EndPoly2 => {
                    self.status = self.prev_status;
                    return PATH_CMD_END_POLY | PATH_FLAGS_CLOSE | PATH_FLAGS_CW;
                }
                S::Stop => cmd = PATH_CMD_STOP,
            }
        }
        cmd
    }
}

const MAX_DASHES: usize = 32;

/// Dash boundaries one dasher crosses (in `calc_dash_start` and along the
/// path) before it refuses: at C++ Agg's speed over ten seconds of dashing
/// one path, which it would carry on for up to hours unless its cell blocks
/// overflow first (e.g. (2e-6, 2e-6) at linewidth 2 overflows after 4.9e7).
pub const MAX_DASH_STEPS: u64 = 1 << 27;

/// Why a dasher stopped where C++ Agg would not have.
pub const DASH_NEVER_ADVANCES: &str = "the dash pattern never advances along the path (Agg loops forever)";
pub const DASH_TOO_MANY_STEPS: &str = "the dash pattern cuts one path into more than 2**27 pieces";

#[derive(Clone, Copy, PartialEq, Eq)]
enum DashStatus { Initial, Ready, Polyline, Stop }

pub struct VcgenDash {
    dashes: [f64; MAX_DASHES],
    total_dash_len: f64,
    num_dashes: usize,
    dash_start: f64,
    curr_dash_start: f64,
    curr_dash: usize,
    curr_rest: f64,
    v1: usize,
    v2: usize,
    src_vertices: VertexSequence,
    closed: u32,
    status: DashStatus,
    src_vertex: usize,
    /// Dash boundaries crossed so far.
    steps: u64,
    /// `curr_rest` at the last wrap to the first dash within this segment (NaN: none).
    wrap_rest: f64,
    /// Set when the dasher stopped short: it then emits nothing more.
    pub refused: Option<&'static str>,
}

impl Default for VcgenDash {
    fn default() -> Self {
        Self { dashes: [0.0; MAX_DASHES], total_dash_len: 0.0, num_dashes: 0, dash_start: 0.0, curr_dash_start: 0.0,
               curr_dash: 0, curr_rest: 0.0, v1: 0, v2: 0, src_vertices: VertexSequence::default(), closed: 0,
               status: DashStatus::Initial, src_vertex: 0, steps: 0, wrap_rest: f64::NAN, refused: None }
    }
}

impl VcgenDash {
    pub fn add_dash(&mut self, dash_len: f64, gap_len: f64) {
        if self.num_dashes < MAX_DASHES {
            self.total_dash_len += dash_len + gap_len;
            self.dashes[self.num_dashes] = dash_len;
            self.dashes[self.num_dashes + 1] = gap_len;
            self.num_dashes += 2;
        }
    }

    pub fn dash_start(&mut self, ds: f64) {
        self.dash_start = ds;
        self.calc_dash_start(ds.abs());
    }

    fn calc_dash_start(&mut self, mut ds: f64) {
        self.curr_dash = 0;
        self.curr_dash_start = 0.0;
        let mut wrap = f64::NAN;
        while ds > 0.0 {
            if ds > self.dashes[self.curr_dash] {
                ds -= self.dashes[self.curr_dash];
                self.curr_dash += 1;
                self.curr_dash_start = 0.0;
                if self.curr_dash >= self.num_dashes {
                    self.curr_dash = 0;
                    // A whole cycle that left `ds` no smaller repeats forever.
                    if ds >= wrap { return self.refuse(DASH_NEVER_ADVANCES); }
                    wrap = ds;
                }
                if !self.step() { return; }
            } else {
                self.curr_dash_start = ds;
                ds = 0.0;
            }
        }
    }

    /// Counts one dash boundary; false (refused) past [`MAX_DASH_STEPS`].
    fn step(&mut self) -> bool {
        self.steps += 1;
        if self.steps > MAX_DASH_STEPS {
            self.refuse(DASH_TOO_MANY_STEPS);
            return false;
        }
        true
    }

    fn refuse(&mut self, why: &'static str) {
        self.refused = Some(why);
        self.status = DashStatus::Stop;
    }
}

impl Generator for VcgenDash {
    fn remove_all(&mut self) {
        self.status = DashStatus::Initial;
        self.src_vertices.remove_all();
        self.closed = 0;
    }

    fn add_vertex(&mut self, x: f64, y: f64, cmd: u32) {
        self.status = DashStatus::Initial;
        if is_move_to(cmd) {
            self.src_vertices.modify_last(VertexDist::new(x, y));
        } else if is_vertex(cmd) {
            self.src_vertices.add(VertexDist::new(x, y));
        } else {
            self.closed = get_close_flag(cmd);
        }
    }

    fn rewind(&mut self, _path_id: u32) {
        if self.status == DashStatus::Initial { self.src_vertices.close(self.closed != 0); }
        self.status = DashStatus::Ready;
        self.src_vertex = 0;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.refused.is_some() { return PATH_CMD_STOP; }
        let mut cmd = PATH_CMD_MOVE_TO;
        while !is_stop(cmd) {
            match self.status {
                DashStatus::Initial | DashStatus::Ready => {
                    if self.status == DashStatus::Initial { self.rewind(0); }
                    if self.num_dashes < 2 || self.src_vertices.len() < 2 {
                        cmd = PATH_CMD_STOP;
                        continue;
                    }
                    self.status = DashStatus::Polyline;
                    self.src_vertex = 1;
                    self.v1 = 0;
                    self.v2 = 1;
                    let v1 = self.src_vertices.get(0);
                    self.curr_rest = v1.dist;
                    self.wrap_rest = f64::NAN;
                    *x = v1.x;
                    *y = v1.y;
                    if self.dash_start >= 0.0 { self.calc_dash_start(self.dash_start); }
                    if self.refused.is_some() { return PATH_CMD_STOP; }
                    return PATH_CMD_MOVE_TO;
                }
                DashStatus::Polyline => {
                    let dash_rest = self.dashes[self.curr_dash] - self.curr_dash_start;
                    let cmd = if self.curr_dash & 1 != 0 { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO };
                    let v1 = self.src_vertices.get(self.v1);
                    let v2 = self.src_vertices.get(self.v2);
                    if self.curr_rest > dash_rest {
                        self.curr_rest -= dash_rest;
                        self.curr_dash += 1;
                        if self.curr_dash >= self.num_dashes {
                            self.curr_dash = 0;
                            // A whole cycle within one segment that left `curr_rest`
                            // no smaller repeats forever (fl(r - d) is monotone in r).
                            if self.curr_rest >= self.wrap_rest {
                                self.refuse(DASH_NEVER_ADVANCES);
                                return PATH_CMD_STOP;
                            }
                            self.wrap_rest = self.curr_rest;
                        }
                        if !self.step() { return PATH_CMD_STOP; }
                        self.curr_dash_start = 0.0;
                        *x = v2.x - (v2.x - v1.x) * self.curr_rest / v1.dist;
                        *y = v2.y - (v2.y - v1.y) * self.curr_rest / v1.dist;
                    } else {
                        self.wrap_rest = f64::NAN;
                        self.curr_dash_start += self.curr_rest;
                        *x = v2.x;
                        *y = v2.y;
                        self.src_vertex += 1;
                        self.v1 = self.v2;
                        self.curr_rest = v2.dist;
                        let n = self.src_vertices.len();
                        if self.closed != 0 {
                            if self.src_vertex > n {
                                self.status = DashStatus::Stop;
                            } else {
                                self.v2 = if self.src_vertex >= n { 0 } else { self.src_vertex };
                            }
                        } else if self.src_vertex >= n {
                            self.status = DashStatus::Stop;
                        } else {
                            self.v2 = self.src_vertex;
                        }
                    }
                    return cmd;
                }
                DashStatus::Stop => cmd = PATH_CMD_STOP,
            }
        }
        PATH_CMD_STOP
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdaptorStatus { Initial, Accumulate, Generate }

/// `conv_adaptor_vcgen<VertexSource, Generator, null_markers>`.
pub struct ConvAdaptorVcgen<S, G> {
    source: S,
    pub generator: G,
    status: AdaptorStatus,
    last_cmd: u32,
    start_x: f64,
    start_y: f64,
}

impl<S: VertexSource, G: Generator + Default> ConvAdaptorVcgen<S, G> {
    pub fn new(source: S) -> Self {
        Self { source, generator: G::default(), status: AdaptorStatus::Initial, last_cmd: 0, start_x: 0.0, start_y: 0.0 }
    }
}

pub type ConvStroke<S> = ConvAdaptorVcgen<S, VcgenStroke>;
pub type ConvDash<S> = ConvAdaptorVcgen<S, VcgenDash>;

impl<S> ConvAdaptorVcgen<S, VcgenStroke> {
    pub fn line_cap(&mut self, lc: LineCap) { self.generator.stroker.line_cap = lc; }
    pub fn line_join(&mut self, lj: LineJoin) { self.generator.stroker.line_join = lj; }
    pub fn width(&mut self, w: f64) { self.generator.stroker.set_width(w); }
    pub fn miter_limit(&mut self, ml: f64) { self.generator.stroker.miter_limit = ml; }
}

impl<S: VertexSource, G: Generator> VertexSource for ConvAdaptorVcgen<S, G> {
    fn rewind(&mut self, path_id: u32) {
        self.source.rewind(path_id);
        self.status = AdaptorStatus::Initial;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        loop {
            match self.status {
                AdaptorStatus::Initial | AdaptorStatus::Accumulate => {
                    if self.status == AdaptorStatus::Initial {
                        self.last_cmd = self.source.vertex(&mut self.start_x, &mut self.start_y);
                        self.status = AdaptorStatus::Accumulate;
                    }
                    if is_stop(self.last_cmd) { return PATH_CMD_STOP; }
                    self.generator.remove_all();
                    self.generator.add_vertex(self.start_x, self.start_y, PATH_CMD_MOVE_TO);
                    loop {
                        let cmd = self.source.vertex(x, y);
                        if is_vertex(cmd) {
                            self.last_cmd = cmd;
                            if is_move_to(cmd) {
                                self.start_x = *x;
                                self.start_y = *y;
                                break;
                            }
                            self.generator.add_vertex(*x, *y, cmd);
                        } else {
                            if is_stop(cmd) {
                                self.last_cmd = PATH_CMD_STOP;
                                break;
                            }
                            if is_end_poly(cmd) {
                                self.generator.add_vertex(*x, *y, cmd);
                                break;
                            }
                        }
                    }
                    self.generator.rewind(0);
                    self.status = AdaptorStatus::Generate;
                }
                AdaptorStatus::Generate => {
                    let cmd = self.generator.vertex(x, y);
                    if is_stop(cmd) {
                        self.status = AdaptorStatus::Accumulate;
                        continue;
                    }
                    return cmd;
                }
            }
        }
    }
}
