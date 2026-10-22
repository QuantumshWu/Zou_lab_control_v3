//! `agg::trans_affine`: products and inverses keep Agg's operand order, so a
//! zero coefficient times an infinity still yields NaN as it does in C++.

pub const AFFINE_EPSILON: f64 = 1e-14;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransAffine {
    pub sx: f64,
    pub shy: f64,
    pub shx: f64,
    pub sy: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Default for TransAffine {
    fn default() -> Self { Self::IDENTITY }
}

impl TransAffine {
    pub const IDENTITY: Self = Self { sx: 1.0, shy: 0.0, shx: 0.0, sy: 1.0, tx: 0.0, ty: 0.0 };

    /// Agg's constructor order: sx, shy, shx, sy, tx, ty.
    pub const fn new(sx: f64, shy: f64, shx: f64, sy: f64, tx: f64, ty: f64) -> Self {
        Self { sx, shy, shx, sy, tx, ty }
    }

    pub const fn scaling(x: f64, y: f64) -> Self { Self::new(x, 0.0, 0.0, y, 0.0, 0.0) }

    pub const fn translation(x: f64, y: f64) -> Self { Self::new(1.0, 0.0, 0.0, 1.0, x, y) }

    pub fn rotation(a: f64) -> Self { Self::new(a.cos(), a.sin(), -a.sin(), a.cos(), 0.0, 0.0) }

    /// `*this *= m`.
    pub fn multiply(&mut self, m: &TransAffine) -> &mut Self {
        let t0 = self.sx * m.sx + self.shy * m.shx;
        let t2 = self.shx * m.sx + self.sy * m.shx;
        let t4 = self.tx * m.sx + self.ty * m.shx + m.tx;
        self.shy = self.sx * m.shy + self.shy * m.sy;
        self.sy = self.shx * m.shy + self.sy * m.sy;
        self.ty = self.tx * m.shy + self.ty * m.sy + m.ty;
        self.sx = t0;
        self.shx = t2;
        self.tx = t4;
        self
    }

    pub fn invert(&mut self) -> &mut Self {
        let d = 1.0 / (self.sx * self.sy - self.shy * self.shx);
        let t0 = self.sy * d;
        self.sy = self.sx * d;
        self.shy = -self.shy * d;
        self.shx = -self.shx * d;
        let t4 = -self.tx * t0 - self.ty * self.shx;
        self.ty = -self.tx * self.shy - self.ty * self.sy;
        self.sx = t0;
        self.tx = t4;
        self
    }

    #[inline(always)]
    pub fn transform(&self, x: &mut f64, y: &mut f64) {
        let tmp = *x;
        *x = tmp * self.sx + *y * self.shx + self.tx;
        *y = tmp * self.shy + *y * self.sy + self.ty;
    }

    pub fn is_equal(&self, m: &TransAffine, epsilon: f64) -> bool {
        (self.sx - m.sx).abs() <= epsilon && (self.shy - m.shy).abs() <= epsilon
            && (self.shx - m.shx).abs() <= epsilon && (self.sy - m.sy).abs() <= epsilon
            && (self.tx - m.tx).abs() <= epsilon && (self.ty - m.ty).abs() <= epsilon
    }
}
