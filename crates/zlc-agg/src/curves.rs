//! `conv_curve` with `curve3_div`/`curve4_div` at Agg's default settings:
//! approximation scale 1, angle tolerance 0 and no cusp limit.
use crate::basics::*;

const CURVE_COLLINEARITY_EPSILON: f64 = 1e-30;
const CURVE_RECURSION_LIMIT: u32 = 32;
/// `0.5 / approximation_scale`, squared.
const DISTANCE_TOLERANCE_SQUARE: f64 = 0.5 * 0.5;

#[derive(Default)]
struct CurvePoints { points: Vec<PointD>, count: usize }

impl CurvePoints {
    fn reset(&mut self) {
        self.points.clear();
        self.count = 0;
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if self.count >= self.points.len() { return PATH_CMD_STOP; }
        let p = self.points[self.count];
        self.count += 1;
        *x = p.x;
        *y = p.y;
        if self.count == 1 { PATH_CMD_MOVE_TO } else { PATH_CMD_LINE_TO }
    }

    fn init3(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) {
        self.points.clear();
        self.points.push(PointD::new(x1, y1));
        self.bezier3(x1, y1, x2, y2, x3, y3, 0);
        self.points.push(PointD::new(x3, y3));
        self.count = 0;
    }

    #[allow(clippy::too_many_arguments)]
    fn bezier3(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, level: u32) {
        if level > CURVE_RECURSION_LIMIT { return; }
        let x12 = (x1 + x2) / 2.0;
        let y12 = (y1 + y2) / 2.0;
        let x23 = (x2 + x3) / 2.0;
        let y23 = (y2 + y3) / 2.0;
        let x123 = (x12 + x23) / 2.0;
        let y123 = (y12 + y23) / 2.0;
        let dx = x3 - x1;
        let dy = y3 - y1;
        let mut d = ((x2 - x3) * dy - (y2 - y3) * dx).abs();
        if d > CURVE_COLLINEARITY_EPSILON {
            if d * d <= DISTANCE_TOLERANCE_SQUARE * (dx * dx + dy * dy) {
                self.points.push(PointD::new(x123, y123));
                return;
            }
        } else {
            let da = dx * dx + dy * dy;
            if da == 0.0 {
                d = calc_sq_distance(x1, y1, x2, y2);
            } else {
                d = ((x2 - x1) * dx + (y2 - y1) * dy) / da;
                if d > 0.0 && d < 1.0 { return; }
                d = if d <= 0.0 { calc_sq_distance(x2, y2, x1, y1) }
                    else if d >= 1.0 { calc_sq_distance(x2, y2, x3, y3) }
                    else { calc_sq_distance(x2, y2, x1 + d * dx, y1 + d * dy) };
            }
            if d < DISTANCE_TOLERANCE_SQUARE {
                self.points.push(PointD::new(x2, y2));
                return;
            }
        }
        self.bezier3(x1, y1, x12, y12, x123, y123, level + 1);
        self.bezier3(x123, y123, x23, y23, x3, y3, level + 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn init4(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, x4: f64, y4: f64) {
        self.points.clear();
        self.points.push(PointD::new(x1, y1));
        self.bezier4(x1, y1, x2, y2, x3, y3, x4, y4, 0);
        self.points.push(PointD::new(x4, y4));
        self.count = 0;
    }

    #[allow(clippy::too_many_arguments)]
    fn bezier4(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, x4: f64, y4: f64, level: u32) {
        if level > CURVE_RECURSION_LIMIT { return; }
        let x12 = (x1 + x2) / 2.0;
        let y12 = (y1 + y2) / 2.0;
        let x23 = (x2 + x3) / 2.0;
        let y23 = (y2 + y3) / 2.0;
        let x34 = (x3 + x4) / 2.0;
        let y34 = (y3 + y4) / 2.0;
        let x123 = (x12 + x23) / 2.0;
        let y123 = (y12 + y23) / 2.0;
        let x234 = (x23 + x34) / 2.0;
        let y234 = (y23 + y34) / 2.0;
        let x1234 = (x123 + x234) / 2.0;
        let y1234 = (y123 + y234) / 2.0;
        let dx = x4 - x1;
        let dy = y4 - y1;
        let mut d2 = ((x2 - x4) * dy - (y2 - y4) * dx).abs();
        let mut d3 = ((x3 - x4) * dy - (y3 - y4) * dx).abs();
        match (u32::from(d2 > CURVE_COLLINEARITY_EPSILON) << 1) + u32::from(d3 > CURVE_COLLINEARITY_EPSILON) {
            0 => {
                let mut k = dx * dx + dy * dy;
                if k == 0.0 {
                    d2 = calc_sq_distance(x1, y1, x2, y2);
                    d3 = calc_sq_distance(x4, y4, x3, y3);
                } else {
                    k = 1.0 / k;
                    let mut da1 = x2 - x1;
                    let mut da2 = y2 - y1;
                    d2 = k * (da1 * dx + da2 * dy);
                    da1 = x3 - x1;
                    da2 = y3 - y1;
                    d3 = k * (da1 * dx + da2 * dy);
                    if d2 > 0.0 && d2 < 1.0 && d3 > 0.0 && d3 < 1.0 { return; }
                    d2 = if d2 <= 0.0 { calc_sq_distance(x2, y2, x1, y1) }
                         else if d2 >= 1.0 { calc_sq_distance(x2, y2, x4, y4) }
                         else { calc_sq_distance(x2, y2, x1 + d2 * dx, y1 + d2 * dy) };
                    d3 = if d3 <= 0.0 { calc_sq_distance(x3, y3, x1, y1) }
                         else if d3 >= 1.0 { calc_sq_distance(x3, y3, x4, y4) }
                         else { calc_sq_distance(x3, y3, x1 + d3 * dx, y1 + d3 * dy) };
                }
                if d2 > d3 {
                    if d2 < DISTANCE_TOLERANCE_SQUARE {
                        self.points.push(PointD::new(x2, y2));
                        return;
                    }
                } else if d3 < DISTANCE_TOLERANCE_SQUARE {
                    self.points.push(PointD::new(x3, y3));
                    return;
                }
            }
            1 => {
                if d3 * d3 <= DISTANCE_TOLERANCE_SQUARE * (dx * dx + dy * dy) {
                    self.points.push(PointD::new(x23, y23));
                    return;
                }
            }
            2 => {
                if d2 * d2 <= DISTANCE_TOLERANCE_SQUARE * (dx * dx + dy * dy) {
                    self.points.push(PointD::new(x23, y23));
                    return;
                }
            }
            _ => {
                if (d2 + d3) * (d2 + d3) <= DISTANCE_TOLERANCE_SQUARE * (dx * dx + dy * dy) {
                    self.points.push(PointD::new(x23, y23));
                    return;
                }
            }
        }
        self.bezier4(x1, y1, x12, y12, x123, y123, x1234, y1234, level + 1);
        self.bezier4(x1234, y1234, x234, y234, x34, y34, x4, y4, level + 1);
    }
}

/// `conv_curve`: CURVE3/CURVE4 control points become line segments from the
/// previous vertex, whatever command produced it.
pub struct ConvCurve<S> { source: S, last_x: f64, last_y: f64, curve3: CurvePoints, curve4: CurvePoints }

impl<S: VertexSource> ConvCurve<S> {
    pub fn new(source: S) -> Self {
        Self { source, last_x: 0.0, last_y: 0.0, curve3: CurvePoints::default(), curve4: CurvePoints::default() }
    }

    pub fn source_mut(&mut self) -> &mut S { &mut self.source }
}

impl<S: VertexSource> VertexSource for ConvCurve<S> {
    fn rewind(&mut self, path_id: u32) {
        self.source.rewind(path_id);
        self.last_x = 0.0;
        self.last_y = 0.0;
        self.curve3.reset();
        self.curve4.reset();
    }

    fn vertex(&mut self, x: &mut f64, y: &mut f64) -> u32 {
        if !is_stop(self.curve3.vertex(x, y)) {
            self.last_x = *x;
            self.last_y = *y;
            return PATH_CMD_LINE_TO;
        }
        if !is_stop(self.curve4.vertex(x, y)) {
            self.last_x = *x;
            self.last_y = *y;
            return PATH_CMD_LINE_TO;
        }
        let mut cmd = self.source.vertex(x, y);
        match cmd {
            PATH_CMD_CURVE3 => {
                let (mut end_x, mut end_y) = (0.0, 0.0);
                self.source.vertex(&mut end_x, &mut end_y);
                self.curve3.init3(self.last_x, self.last_y, *x, *y, end_x, end_y);
                self.curve3.vertex(x, y);
                self.curve3.vertex(x, y);
                cmd = PATH_CMD_LINE_TO;
            }
            PATH_CMD_CURVE4 => {
                let (mut ct2_x, mut ct2_y, mut end_x, mut end_y) = (0.0, 0.0, 0.0, 0.0);
                self.source.vertex(&mut ct2_x, &mut ct2_y);
                self.source.vertex(&mut end_x, &mut end_y);
                self.curve4.init4(self.last_x, self.last_y, *x, *y, ct2_x, ct2_y, end_x, end_y);
                self.curve4.vertex(x, y);
                self.curve4.vertex(x, y);
                cmd = PATH_CMD_LINE_TO;
            }
            _ => {}
        }
        self.last_x = *x;
        self.last_y = *y;
        cmd
    }
}
