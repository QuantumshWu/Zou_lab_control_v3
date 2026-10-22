//! Degenerate dash patterns: where C++ Agg's dasher never stops (a cycle that
//! does not advance along the path) the stroke is refused at once; where it
//! stops, it stops as C++ Agg does (matplotlib 3.10.8 checked: a collection
//! dashed (0, 0), (2, -3) or offset 5 into (0, 0) at linewidth 2 hangs its
//! draw; non-antialiased (0, 0) draws half-pixel dashes; (1e-6, 1e-6) raises
//! "Exceeded cell block limit").
use std::time::{Duration, Instant};

use zlc_agg::basics::RectD;
use zlc_agg::color::Rgba;
use zlc_agg::path::{PathData, SnapMode};
use zlc_agg::trans_affine::TransAffine;
use zlc_agg::vcgen::{LineCap, LineJoin, DASH_NEVER_ADVANCES, DASH_TOO_MANY_STEPS};
use zlc_agg::{AggError, ClipPath, Dashes, Gc, RendererAgg, SketchParams};

/// The sine of a 400x300, dpi-100 axes at (0.1, 0.1, 0.8, 0.8), x 0..10, y -2..2.
fn sine() -> Vec<[f64; 2]> {
    (0..50).map(|i| {
        let x = 10.0 * f64::from(i) / 49.0;
        [40.0 + 32.0 * x, 30.0 + 60.0 * (x.sin() + 2.0)]
    }).collect()
}

fn gc(linewidth: f64, isaa: bool, offset: f64, pairs: &[(f64, f64)]) -> Gc<'static> {
    Gc {
        linewidth, alpha: 1.0, forced_alpha: false, color: Rgba::new(0.0, 0.0, 1.0, 1.0), isaa,
        cap: LineCap::Butt, join: LineJoin::Round, cliprect: RectD::default(),
        clippath: ClipPath { path: PathData::EMPTY, trans: TransAffine::IDENTITY },
        dashes: Dashes { offset, pairs: pairs.to_vec() }, snap_mode: SnapMode::Auto, hatchpath: PathData::EMPTY,
        hatch_color: Rgba::new(0.0, 0.0, 0.0, 1.0), hatch_linewidth: 1.0, sketch: SketchParams::default(),
    }
}

/// One dashed stroke of `vertices` on a fresh canvas: the result, the time and
/// whether the canvas is still blank.
fn stroke(vertices: &[[f64; 2]], gc: &Gc<'_>) -> (Result<(), AggError>, Duration, bool) {
    let mut r = RendererAgg::new(400, 300, 100.0).unwrap();
    let blank = r.pixels.clone();
    let path = PathData { vertices, codes: None, should_simplify: false, simplify_threshold: 1.0 / 9.0, id: 1 };
    let t = Instant::now();
    let result = r.draw_path(gc, path, TransAffine::IDENTITY, Rgba::new(0.0, 0.0, 0.0, 0.0));
    (result, t.elapsed(), r.pixels == blank)
}

fn refused(result: &Result<(), AggError>, why: &str) -> bool { *result == Err(AggError::Value(why.into())) }

#[test]
fn a_pattern_that_never_advances_is_refused_at_once() {
    let line = sine();
    for (offset, pairs, isaa) in [
        (0.0, vec![(0.0, 0.0)], true),
        (5.0, vec![(0.0, 0.0)], true),          // stuck in calc_dash_start
        (0.0, vec![(2.0, -3.0)], true),         // the cycle grows the rest
        (0.0, vec![(1.0, 2.0), (-5.0, 1.0)], true),
        (f64::INFINITY, vec![(2.0, 3.0)], true), // an infinite offset never runs out
        (0.0, vec![(f64::NEG_INFINITY, 3.0)], true),
        (0.0, vec![(1e-30, 1e-30)], true),      // r - 1e-30 == r: no progress in floats
        (0.0, vec![(-2.0, 1.0)], false),        // non-antialiased (int)(-2)+0.5, 1.5: sum 0
    ] {
        let (result, elapsed, blank) = stroke(&line, &gc(2.0, isaa, offset, &pairs));
        assert!(refused(&result, DASH_NEVER_ADVANCES), "{offset} {pairs:?} {isaa}: {result:?}");
        assert!(elapsed < Duration::from_secs(1), "{offset} {pairs:?}: {elapsed:?}");
        assert!(blank, "{offset} {pairs:?}: drew");
    }
}

#[test]
fn degenerate_patterns_that_stop_stop_as_in_agg() {
    let line = sine();
    // Non-antialiased zeros become half-pixel dashes; NaN and infinite dashes
    // end the cycle; a negative dash with a positive cycle advances.
    for (offset, pairs, isaa) in [
        (0.0, vec![(0.0, 0.0)], false),
        (0.0, vec![(f64::NAN, 2.0)], true),
        (0.0, vec![(3.0, f64::INFINITY)], true),
        (f64::NAN, vec![(3.0, 2.0)], true),
        (0.0, vec![(-2.0, 4.0)], true),
        (0.0, vec![(0.0, 4.0)], true),
    ] {
        let (result, elapsed, _) = stroke(&line, &gc(2.0, isaa, offset, &pairs));
        assert_eq!(result, Ok(()), "{offset} {pairs:?} {isaa}");
        assert!(elapsed < Duration::from_secs(1), "{offset} {pairs:?}: {elapsed:?}");
    }
    // Matplotlib: "Exceeded cell block limit" for (1e-6, 1e-6) at linewidth 2.
    let (result, _, _) = stroke(&line, &gc(2.0, true, 0.0, &[(2e-6, 2e-6)]));
    assert_eq!(result, Err(AggError::Overflow("Exceeded cell block limit".into())));
}

#[test]
fn a_pattern_past_the_step_limit_is_refused() {
    // Zero-length dashes add no cells, so Agg would cross ~1e12 boundaries.
    let (result, elapsed, blank) = stroke(&sine(), &gc(2.0, true, 0.0, &[(0.0, 1e-9)]));
    assert!(refused(&result, DASH_TOO_MANY_STEPS), "{result:?}");
    assert!(elapsed < Duration::from_secs(20), "{elapsed:?}");
    assert!(blank);
}
