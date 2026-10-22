//! `rasterizer_scanline_aa<rasterizer_sl_clip_dbl>` with 24.8 cells.
//!
//! All state persists between paths exactly as in Agg: a rasterizer whose
//! last sweep found no cells is not sorted, so its bounds and clipper state
//! carry into the next path.
use crate::basics::*;

#[derive(Clone, Copy, Debug)]
pub struct Cell { pub x: i32, pub y: i32, pub cover: i32, pub area: i32 }

const INITIAL_CELL: Cell = Cell { x: 0x7FFFFFFF, y: 0x7FFFFFFF, cover: 0, area: 0 };
const CELL_BLOCK_SIZE: usize = 1 << 12;

/// The receiver of one swept scanline.
pub trait ScanlineSink {
    fn reset(&mut self, min_x: i32, max_x: i32);
    fn reset_spans(&mut self);
    fn add_cell(&mut self, x: i32, cover: u32);
    fn add_span(&mut self, x: i32, len: u32, cover: u32);
    fn num_spans(&self) -> usize;
    fn finalize(&mut self, y: i32);
}

pub struct CellsAa {
    cells: Vec<Cell>,
    /// Blocks allocated by any earlier path; Agg's limit test reads this.
    num_blocks: usize,
    cell_block_limit: usize,
    curr_cell: Cell,
    sorted_cells: Vec<Cell>,
    sorted_y: Vec<(u32, u32)>,
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    sorted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellBlockLimit;

impl CellsAa {
    pub fn new(cell_block_limit: usize) -> Self {
        Self { cells: Vec::new(), num_blocks: 0, cell_block_limit, curr_cell: INITIAL_CELL,
               sorted_cells: Vec::new(), sorted_y: Vec::new(), min_x: 0x7FFFFFFF, min_y: 0x7FFFFFFF,
               max_x: -0x7FFFFFFF, max_y: -0x7FFFFFFF, sorted: false }
    }

    pub fn reset(&mut self) {
        self.cells.clear();
        self.curr_cell = INITIAL_CELL;
        self.sorted = false;
        self.min_x = 0x7FFFFFFF;
        self.min_y = 0x7FFFFFFF;
        self.max_x = -0x7FFFFFFF;
        self.max_y = -0x7FFFFFFF;
    }

    #[inline(always)]
    fn add_curr_cell(&mut self) -> Result<(), CellBlockLimit> {
        if self.curr_cell.area | self.curr_cell.cover != 0 {
            if self.cells.len() & (CELL_BLOCK_SIZE - 1) == 0 {
                if self.num_blocks >= self.cell_block_limit { return Err(CellBlockLimit); }
                let block = self.cells.len() / CELL_BLOCK_SIZE;
                if block >= self.num_blocks { self.num_blocks += 1; }
            }
            self.cells.push(self.curr_cell);
        }
        Ok(())
    }

    #[inline(always)]
    fn set_curr_cell(&mut self, x: i32, y: i32) -> Result<(), CellBlockLimit> {
        if self.curr_cell.x != x || self.curr_cell.y != y {
            self.add_curr_cell()?;
            self.curr_cell = Cell { x, y, cover: 0, area: 0 };
        }
        Ok(())
    }

    #[inline(always)]
    fn render_hline(&mut self, ey: i32, x1: i32, mut y1: i32, x2: i32, y2: i32) -> Result<(), CellBlockLimit> {
        let mut ex1 = x1 >> POLY_SUBPIXEL_SHIFT;
        let ex2 = x2 >> POLY_SUBPIXEL_SHIFT;
        let fx1 = x1 & POLY_SUBPIXEL_MASK;
        let fx2 = x2 & POLY_SUBPIXEL_MASK;
        if y1 == y2 { return self.set_curr_cell(ex2, ey); }
        if ex1 == ex2 {
            let delta = y2 - y1;
            self.curr_cell.cover = self.curr_cell.cover.wrapping_add(delta);
            self.curr_cell.area = self.curr_cell.area.wrapping_add((fx1 + fx2) * delta);
            return Ok(());
        }
        let mut p = (POLY_SUBPIXEL_SCALE - fx1) * (y2 - y1);
        let mut first = POLY_SUBPIXEL_SCALE;
        let mut incr = 1;
        let mut dx = x2.wrapping_sub(x1);
        if dx < 0 {
            p = fx1 * (y2 - y1);
            first = 0;
            incr = -1;
            dx = dx.wrapping_neg();
        }
        let mut delta = p / dx;
        let mut modulo = p % dx;
        if modulo < 0 { delta -= 1; modulo += dx; }
        self.curr_cell.cover = self.curr_cell.cover.wrapping_add(delta);
        self.curr_cell.area = self.curr_cell.area.wrapping_add((fx1 + first) * delta);
        ex1 += incr;
        self.set_curr_cell(ex1, ey)?;
        y1 += delta;
        if ex1 != ex2 {
            let p = POLY_SUBPIXEL_SCALE * (y2 - y1 + delta);
            let mut lift = p / dx;
            let mut rem = p % dx;
            if rem < 0 { lift -= 1; rem += dx; }
            modulo -= dx;
            while ex1 != ex2 {
                delta = lift;
                modulo += rem;
                if modulo >= 0 { modulo -= dx; delta += 1; }
                self.curr_cell.cover = self.curr_cell.cover.wrapping_add(delta);
                self.curr_cell.area = self.curr_cell.area.wrapping_add(POLY_SUBPIXEL_SCALE * delta);
                y1 += delta;
                ex1 += incr;
                self.set_curr_cell(ex1, ey)?;
            }
        }
        delta = y2 - y1;
        self.curr_cell.cover = self.curr_cell.cover.wrapping_add(delta);
        self.curr_cell.area = self.curr_cell.area.wrapping_add((fx2 + POLY_SUBPIXEL_SCALE - first) * delta);
        Ok(())
    }

    pub fn line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) -> Result<(), CellBlockLimit> {
        const DX_LIMIT: i32 = 16384 << POLY_SUBPIXEL_SHIFT;
        let dx = x2.wrapping_sub(x1);
        if dx >= DX_LIMIT || dx <= -DX_LIMIT {
            let cx = (x1 >> 1) + (x2 >> 1) + ((x1 & 1) & (x2 & 1));
            let cy = (y1 >> 1) + (y2 >> 1) + ((y1 & 1) & (y2 & 1));
            self.line(x1, y1, cx, cy)?;
            return self.line(cx, cy, x2, y2);
        }
        let mut dy = y2.wrapping_sub(y1);
        let ex1 = x1 >> POLY_SUBPIXEL_SHIFT;
        let ex2 = x2 >> POLY_SUBPIXEL_SHIFT;
        let mut ey1 = y1 >> POLY_SUBPIXEL_SHIFT;
        let ey2 = y2 >> POLY_SUBPIXEL_SHIFT;
        let fy1 = y1 & POLY_SUBPIXEL_MASK;
        let fy2 = y2 & POLY_SUBPIXEL_MASK;

        if ex1 < self.min_x { self.min_x = ex1; }
        if ex1 > self.max_x { self.max_x = ex1; }
        if ey1 < self.min_y { self.min_y = ey1; }
        if ey1 > self.max_y { self.max_y = ey1; }
        if ex2 < self.min_x { self.min_x = ex2; }
        if ex2 > self.max_x { self.max_x = ex2; }
        if ey2 < self.min_y { self.min_y = ey2; }
        if ey2 > self.max_y { self.max_y = ey2; }

        self.set_curr_cell(ex1, ey1)?;
        if ey1 == ey2 { return self.render_hline(ey1, x1, fy1, x2, fy2); }

        let mut incr = 1;
        if dx == 0 {
            let ex = x1 >> POLY_SUBPIXEL_SHIFT;
            let two_fx = (x1 - (ex << POLY_SUBPIXEL_SHIFT)) << 1;
            let mut first = POLY_SUBPIXEL_SCALE;
            if dy < 0 { first = 0; incr = -1; }
            let mut delta = first - fy1;
            self.curr_cell.cover = self.curr_cell.cover.wrapping_add(delta);
            self.curr_cell.area = self.curr_cell.area.wrapping_add(two_fx * delta);
            ey1 += incr;
            self.set_curr_cell(ex, ey1)?;
            delta = first + first - POLY_SUBPIXEL_SCALE;
            let area = two_fx * delta;
            while ey1 != ey2 {
                self.curr_cell.cover = delta;
                self.curr_cell.area = area;
                ey1 += incr;
                self.set_curr_cell(ex, ey1)?;
            }
            delta = fy2 - POLY_SUBPIXEL_SCALE + first;
            self.curr_cell.cover = self.curr_cell.cover.wrapping_add(delta);
            self.curr_cell.area = self.curr_cell.area.wrapping_add(two_fx * delta);
            return Ok(());
        }

        let mut p = (POLY_SUBPIXEL_SCALE - fy1).wrapping_mul(dx);
        let mut first = POLY_SUBPIXEL_SCALE;
        if dy < 0 {
            p = fy1.wrapping_mul(dx);
            first = 0;
            incr = -1;
            dy = dy.wrapping_neg();
        }
        let mut delta = p / dy;
        let mut modulo = p % dy;
        if modulo < 0 { delta -= 1; modulo += dy; }
        let mut x_from = x1.wrapping_add(delta);
        self.render_hline(ey1, x1, fy1, x_from, first)?;
        ey1 += incr;
        self.set_curr_cell(x_from >> POLY_SUBPIXEL_SHIFT, ey1)?;
        if ey1 != ey2 {
            let p = POLY_SUBPIXEL_SCALE.wrapping_mul(dx);
            let mut lift = p / dy;
            let mut rem = p % dy;
            if rem < 0 { lift -= 1; rem += dy; }
            modulo -= dy;
            while ey1 != ey2 {
                delta = lift;
                modulo += rem;
                if modulo >= 0 { modulo -= dy; delta += 1; }
                let x_to = x_from.wrapping_add(delta);
                self.render_hline(ey1, x_from, POLY_SUBPIXEL_SCALE - first, x_to, first)?;
                x_from = x_to;
                ey1 += incr;
                self.set_curr_cell(x_from >> POLY_SUBPIXEL_SHIFT, ey1)?;
            }
        }
        self.render_hline(ey1, x_from, POLY_SUBPIXEL_SCALE - first, x2, fy2)
    }

    pub fn sort_cells(&mut self) -> Result<(), CellBlockLimit> {
        if self.sorted { return Ok(()); }
        self.add_curr_cell()?;
        self.curr_cell = INITIAL_CELL;
        self.curr_cell.cover = 0;
        self.curr_cell.area = 0;
        if self.cells.is_empty() { return Ok(()); }
        let rows = (i64::from(self.max_y) - i64::from(self.min_y) + 1) as usize;
        self.sorted_y.clear();
        self.sorted_y.resize(rows, (0, 0));
        for cell in &self.cells {
            self.sorted_y[(cell.y - self.min_y) as usize].0 += 1;
        }
        let mut start = 0u32;
        for row in self.sorted_y.iter_mut() {
            let count = row.0;
            row.0 = start;
            start += count;
        }
        self.sorted_cells.clear();
        self.sorted_cells.resize(self.cells.len(), INITIAL_CELL);
        for cell in &self.cells {
            let row = &mut self.sorted_y[(cell.y - self.min_y) as usize];
            self.sorted_cells[(row.0 + row.1) as usize] = *cell;
            row.1 += 1;
        }
        // Cells of one x are summed in any order, so an unstable sort by x
        // sweeps the same coverage as Agg's own quicksort.
        for &(start, num) in &self.sorted_y {
            if num > 1 {
                self.sorted_cells[start as usize..(start + num) as usize].sort_unstable_by_key(|cell| cell.x);
            }
        }
        self.sorted = true;
        Ok(())
    }

    pub fn total_cells(&self) -> usize { self.cells.len() }
    pub fn sorted(&self) -> bool { self.sorted }
    pub fn min_x(&self) -> i32 { self.min_x }
    pub fn min_y(&self) -> i32 { self.min_y }
    pub fn max_x(&self) -> i32 { self.max_x }
    pub fn max_y(&self) -> i32 { self.max_y }

    #[inline(always)]
    fn scanline_cells(&self, y: i32) -> &[Cell] {
        let (start, num) = self.sorted_y[(y - self.min_y) as usize];
        &self.sorted_cells[start as usize..(start + num) as usize]
    }
}

#[inline(always)]
fn clipping_flags(x: f64, y: f64, b: &RectD) -> u32 {
    u32::from(x > b.x2) | (u32::from(y > b.y2) << 1) | (u32::from(x < b.x1) << 2) | (u32::from(y < b.y1) << 3)
}

#[inline(always)]
fn clipping_flags_y(y: f64, b: &RectD) -> u32 { (u32::from(y > b.y2) << 1) | (u32::from(y < b.y1) << 3) }

#[inline(always)]
fn xi(v: f64) -> i32 { iround(v * f64::from(POLY_SUBPIXEL_SCALE)) }

#[inline(always)]
fn mul_div(a: f64, b: f64, c: f64) -> f64 { a * b / c }

/// `rasterizer_sl_clip<ras_conv_dbl>`.
struct ClipDbl { clip_box: RectD, x1: f64, y1: f64, f1: u32, clipping: bool }

impl ClipDbl {
    fn line_clip_y(&self, ras: &mut CellsAa, x1: f64, y1: f64, x2: f64, y2: f64, f1: u32, f2: u32)
        -> Result<(), CellBlockLimit> {
        let f1 = f1 & 10;
        let f2 = f2 & 10;
        if f1 | f2 == 0 { return ras.line(xi(x1), xi(y1), xi(x2), xi(y2)); }
        if f1 == f2 { return Ok(()); }
        let b = &self.clip_box;
        let (mut tx1, mut ty1, mut tx2, mut ty2) = (x1, y1, x2, y2);
        if f1 & 8 != 0 { tx1 = x1 + mul_div(b.y1 - y1, x2 - x1, y2 - y1); ty1 = b.y1; }
        if f1 & 2 != 0 { tx1 = x1 + mul_div(b.y2 - y1, x2 - x1, y2 - y1); ty1 = b.y2; }
        if f2 & 8 != 0 { tx2 = x1 + mul_div(b.y1 - y1, x2 - x1, y2 - y1); ty2 = b.y1; }
        if f2 & 2 != 0 { tx2 = x1 + mul_div(b.y2 - y1, x2 - x1, y2 - y1); ty2 = b.y2; }
        ras.line(xi(tx1), xi(ty1), xi(tx2), xi(ty2))
    }

    fn move_to(&mut self, x1: f64, y1: f64) {
        self.x1 = x1;
        self.y1 = y1;
        if self.clipping { self.f1 = clipping_flags(x1, y1, &self.clip_box); }
    }

    fn line_to(&mut self, ras: &mut CellsAa, x2: f64, y2: f64) -> Result<(), CellBlockLimit> {
        if self.clipping {
            let f2 = clipping_flags(x2, y2, &self.clip_box);
            if (self.f1 & 10) == (f2 & 10) && (self.f1 & 10) != 0 {
                self.x1 = x2;
                self.y1 = y2;
                self.f1 = f2;
                return Ok(());
            }
            let (x1, y1, f1) = (self.x1, self.y1, self.f1);
            let b = self.clip_box;
            match ((f1 & 5) << 1) | (f2 & 5) {
                0 => self.line_clip_y(ras, x1, y1, x2, y2, f1, f2)?,
                1 => {
                    let y3 = y1 + mul_div(b.x2 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &b);
                    self.line_clip_y(ras, x1, y1, b.x2, y3, f1, f3)?;
                    self.line_clip_y(ras, b.x2, y3, b.x2, y2, f3, f2)?;
                }
                2 => {
                    let y3 = y1 + mul_div(b.x2 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &b);
                    self.line_clip_y(ras, b.x2, y1, b.x2, y3, f1, f3)?;
                    self.line_clip_y(ras, b.x2, y3, x2, y2, f3, f2)?;
                }
                3 => self.line_clip_y(ras, b.x2, y1, b.x2, y2, f1, f2)?,
                4 => {
                    let y3 = y1 + mul_div(b.x1 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &b);
                    self.line_clip_y(ras, x1, y1, b.x1, y3, f1, f3)?;
                    self.line_clip_y(ras, b.x1, y3, b.x1, y2, f3, f2)?;
                }
                6 => {
                    let y3 = y1 + mul_div(b.x2 - x1, y2 - y1, x2 - x1);
                    let y4 = y1 + mul_div(b.x1 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &b);
                    let f4 = clipping_flags_y(y4, &b);
                    self.line_clip_y(ras, b.x2, y1, b.x2, y3, f1, f3)?;
                    self.line_clip_y(ras, b.x2, y3, b.x1, y4, f3, f4)?;
                    self.line_clip_y(ras, b.x1, y4, b.x1, y2, f4, f2)?;
                }
                8 => {
                    let y3 = y1 + mul_div(b.x1 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &b);
                    self.line_clip_y(ras, b.x1, y1, b.x1, y3, f1, f3)?;
                    self.line_clip_y(ras, b.x1, y3, x2, y2, f3, f2)?;
                }
                9 => {
                    let y3 = y1 + mul_div(b.x1 - x1, y2 - y1, x2 - x1);
                    let y4 = y1 + mul_div(b.x2 - x1, y2 - y1, x2 - x1);
                    let f3 = clipping_flags_y(y3, &b);
                    let f4 = clipping_flags_y(y4, &b);
                    self.line_clip_y(ras, b.x1, y1, b.x1, y3, f1, f3)?;
                    self.line_clip_y(ras, b.x1, y3, b.x2, y4, f3, f4)?;
                    self.line_clip_y(ras, b.x2, y4, b.x2, y2, f4, f2)?;
                }
                12 => self.line_clip_y(ras, b.x1, y1, b.x1, y2, f1, f2)?,
                _ => {}
            }
            self.f1 = f2;
        } else {
            ras.line(xi(self.x1), xi(self.y1), xi(x2), xi(y2))?;
        }
        self.x1 = x2;
        self.y1 = y2;
        Ok(())
    }
}

const STATUS_INITIAL: u32 = 0;
const STATUS_MOVE_TO: u32 = 1;
const STATUS_LINE_TO: u32 = 2;
const STATUS_CLOSED: u32 = 3;

/// Matplotlib's `rasterizer_scanline_aa<rasterizer_sl_clip_dbl>`: non-zero
/// filling, automatic close and linear gamma, the settings it never changes.
pub struct RasterizerScanlineAa {
    outline: CellsAa,
    clipper: ClipDbl,
    start_x: f64,
    start_y: f64,
    status: u32,
    scan_y: i32,
}

impl RasterizerScanlineAa {
    pub fn new(cell_block_limit: usize) -> Self {
        Self { outline: CellsAa::new(cell_block_limit),
               clipper: ClipDbl { clip_box: RectD::new(0.0, 0.0, 0.0, 0.0), x1: 0.0, y1: 0.0, f1: 0, clipping: false },
               start_x: 0.0, start_y: 0.0, status: STATUS_INITIAL, scan_y: 0 }
    }

    pub fn reset(&mut self) {
        self.outline.reset();
        self.status = STATUS_INITIAL;
    }

    pub fn reset_clipping(&mut self) {
        self.reset();
        self.clipper.clipping = false;
    }

    pub fn clip_box(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) {
        self.reset();
        let mut b = RectD::new(x1, y1, x2, y2);
        b.normalize();
        self.clipper.clip_box = b;
        self.clipper.clipping = true;
    }

    pub fn close_polygon(&mut self) -> Result<(), CellBlockLimit> {
        if self.status == STATUS_LINE_TO {
            self.clipper.line_to(&mut self.outline, self.start_x, self.start_y)?;
            self.status = STATUS_CLOSED;
        }
        Ok(())
    }

    pub fn move_to_d(&mut self, x: f64, y: f64) -> Result<(), CellBlockLimit> {
        if self.outline.sorted() { self.reset(); }
        self.close_polygon()?;
        self.start_x = x;
        self.start_y = y;
        self.clipper.move_to(x, y);
        self.status = STATUS_MOVE_TO;
        Ok(())
    }

    pub fn line_to_d(&mut self, x: f64, y: f64) -> Result<(), CellBlockLimit> {
        self.clipper.line_to(&mut self.outline, x, y)?;
        self.status = STATUS_LINE_TO;
        Ok(())
    }

    #[inline]
    pub fn add_vertex(&mut self, x: f64, y: f64, cmd: u32) -> Result<(), CellBlockLimit> {
        if is_move_to(cmd) { self.move_to_d(x, y) }
        else if is_vertex(cmd) { self.line_to_d(x, y) }
        else if is_close(cmd) { self.close_polygon() }
        else { Ok(()) }
    }

    pub fn add_path<V: VertexSource + ?Sized>(&mut self, vs: &mut V, path_id: u32) -> Result<(), CellBlockLimit> {
        let mut x = 0.0;
        let mut y = 0.0;
        vs.rewind(path_id);
        if self.outline.sorted() { self.reset(); }
        loop {
            let cmd = vs.vertex(&mut x, &mut y);
            if is_stop(cmd) { break; }
            self.add_vertex(x, y, cmd)?;
        }
        Ok(())
    }

    pub fn min_x(&self) -> i32 { self.outline.min_x() }
    pub fn min_y(&self) -> i32 { self.outline.min_y() }
    pub fn max_x(&self) -> i32 { self.outline.max_x() }
    pub fn max_y(&self) -> i32 { self.outline.max_y() }

    pub fn rewind_scanlines(&mut self) -> Result<bool, CellBlockLimit> {
        self.close_polygon()?;
        self.outline.sort_cells()?;
        if self.outline.total_cells() == 0 { return Ok(false); }
        self.scan_y = self.outline.min_y();
        Ok(true)
    }

    #[inline(always)]
    fn calculate_alpha(area: i32) -> u32 {
        let mut cover = area >> (POLY_SUBPIXEL_SHIFT * 2 + 1 - 8);
        if cover < 0 { cover = cover.wrapping_neg(); }
        if cover > 255 { cover = 255; }
        cover as u32
    }

    pub fn sweep_scanline<S: ScanlineSink + ?Sized>(&mut self, sl: &mut S) -> bool {
        loop {
            if self.scan_y > self.outline.max_y() { return false; }
            sl.reset_spans();
            let cells = self.outline.scanline_cells(self.scan_y);
            let mut index = 0;
            let mut cover = 0i32;
            while index < cells.len() {
                let mut cur = cells[index];
                let mut x = cur.x;
                let mut area = cur.area;
                cover = cover.wrapping_add(cur.cover);
                index += 1;
                while index < cells.len() {
                    cur = cells[index];
                    if cur.x != x { break; }
                    area = area.wrapping_add(cur.area);
                    cover = cover.wrapping_add(cur.cover);
                    index += 1;
                }
                if area != 0 {
                    let alpha = Self::calculate_alpha((cover << (POLY_SUBPIXEL_SHIFT + 1)).wrapping_sub(area));
                    if alpha != 0 { sl.add_cell(x, alpha); }
                    x += 1;
                }
                if index < cells.len() && cur.x > x {
                    let alpha = Self::calculate_alpha(cover << (POLY_SUBPIXEL_SHIFT + 1));
                    if alpha != 0 { sl.add_span(x, (cur.x - x) as u32, alpha); }
                }
            }
            if sl.num_spans() != 0 { break; }
            self.scan_y += 1;
        }
        sl.finalize(self.scan_y);
        self.scan_y += 1;
        true
    }
}
