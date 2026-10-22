//! How long zlc-render's stroke kernels take on a panel-sized canvas.
use zlc_render::strokes::{raster_polylines, PolylineBatch};
use zlc_render::Canvas;

fn main() {
    let (w, h) = (490usize, 357usize);
    let mut pixels = vec![255u8; w * h * 4];
    for n in [40usize, 3000, 30000] {
        let vertices: Vec<[f64; 2]> = (0..n).map(|i| [60.0 + 300.0 * i as f64 / n as f64, 100.0 + 80.0 * ((i as f64) * 0.01).sin()]).collect();
        let offsets = [0i64, n as i64];
        let batch = PolylineBatch { vertices: &vertices, offsets: &offsets, colours: &[[128, 128, 128, 204]], widths: &[2.9], clips: &[[60, 20, 400, 300]] };
        let mut best = f64::INFINITY;
        for _ in 0..50 {
            let t = std::time::Instant::now();
            let mut canvas = Canvas::new(&mut pixels, w, h).unwrap();
            raster_polylines(&batch, &mut canvas).unwrap();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("raster_polylines {n} points: best {best:.3} ms (threads {})", std::thread::available_parallelism().unwrap());
    }
}
