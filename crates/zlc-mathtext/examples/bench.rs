//! Native per-formula timing: `cargo run --release --example bench -- MATH_FONTS_DIR [SIZE DPI]`.
//! Uncached = parse + ship + raster after `clear_cache()` (faces and glyph metrics
//! loaded); new-string = a live readout whose number changes every call; cached =
//! the raster cache hit a redraw of the same string takes. Medians in µs.

use std::path::Path;
use std::time::Instant;

use zlc_mathtext::{Config, MathText, Request};

const HELVETICA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../science/src/zlc_plot/assets/helvetica-light-587ebe5a59211.ttf");

const FORMULAS: &[&str] = &[
    r"$A$", r"$\sigma$", r"$x_0$", r"$\mathrm{FWHM}$",
    r"$A$ = 13.3 ± 0.2869 count", r"$\sigma$ = 17.14 ± 0.1447 pixel", r"$\mathrm{FWHM}$ = 0.3847 ± 0.007384 MHz",
    r"$f(x)=A e^{-\frac{1}{2}((x-x_0)/\sigma)^2}+B$",
    r"$f(t)=A\sin(2\pi f (t-t_0)+\phi)e^{-(t-t_0)/\tau}+B$",
    r"$f(x)=H\frac{(\mathrm{FWHM}/2)^2}{(x-x_0)^2+(\mathrm{FWHM}/2)^2}+B$",
    r"$f(x)=\frac{Nw}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}((x-x_0)/\sigma)^2}$",
    r"$f(x)=Nw\left[\frac{1-r}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}((x-x_0)/\sigma)^2}+\frac{r}{\sigma_B\sqrt{2\pi}}e^{-\frac{1}{2}((x-x_0-\delta)/\sigma_B)^2}\right]$",
    r"$f(x)=Nw\,P(x;\lambda,\sigma),\ P(x;\lambda,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}\int_0^{\infty}\frac{\lambda^u e^{-\lambda}}{\Gamma(u+1)}\,e^{-\frac{1}{2}((x-u)/\sigma)^2}\,du$",
];

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn time(repeats: usize, mut work: impl FnMut(usize)) -> f64 {
    median((0..repeats).map(|i| { let start = Instant::now(); work(i); start.elapsed().as_secs_f64() * 1e6 }).collect())
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let fonts = arguments.first().expect("MATH_FONTS_DIR");
    let size: f64 = arguments.get(1).map_or(3.25, |v| v.parse().unwrap());
    let dpi: f64 = arguments.get(2).map_or(525.0, |v| v.parse().unwrap());
    let mut engine = MathText::new(Config::new(fonts));
    let font_file = Path::new(HELVETICA);
    let first = Instant::now();
    engine.raster(&Request { text: FORMULAS[7], font_file, size, dpi }, true).unwrap();
    println!("first raster (opens faces): {:.0} µs", first.elapsed().as_secs_f64() * 1e6);
    println!("{:60} {:>10} {:>10} {:>10}", "formula", "uncached", "metrics", "cached");
    for &text in FORMULAS {
        let request = Request { text, font_file, size, dpi };
        engine.raster(&request, true).unwrap();
        let uncached = time(400, |_| { engine.clear_cache(); engine.raster(&request, true).unwrap(); });
        let metrics = time(400, |_| { engine.clear_cache(); engine.metrics(&request).unwrap(); });
        engine.raster(&request, true).unwrap();
        let cached = time(4000, |_| { engine.raster(&request, true).unwrap(); });
        let label: String = if text.chars().count() > 58 { text.chars().take(55).chain("...".chars()).collect() } else { text.into() };
        println!("{label:60} {uncached:10.1} {metrics:10.1} {cached:10.2}");
    }
    let readout = time(2000, |i| {
        let text = format!(r"$\sigma$ = {:.4} ± 0.1447 pixel", 17.0 + i as f64 * 1e-4);
        engine.raster(&Request { text: &text, font_file, size, dpi }, true).unwrap();
    });
    println!("live readout, a new value each call: {readout:.1} µs");
}
