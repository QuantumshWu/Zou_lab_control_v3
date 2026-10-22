//! Values Matplotlib 3.10.8's `ft2font` returns for the bundled Helvetica Light
//! (the full comparison is `research/text-parity/parity.py`).
use zlc_text::{FT2Font, FT2Image};

const HELVETICA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../science/src/zlc_plot/assets/helvetica-light-587ebe5a59211.ttf");
const FORCE_AUTOHINT: i32 = 1 << 5;

fn font() -> FT2Font { FT2Font::new(std::fs::read(HELVETICA).unwrap(), 8, Vec::new(), 0).unwrap() }

fn no_warning(_: std::ffi::c_ulong, _: &[u8]) -> Result<(), zlc_text::Error> { panic!("unexpected missing glyph") }

#[test]
fn live_readout_string() {
    let mut font = font();
    font.set_size(16.0, 180.0).unwrap();
    let mut xys = Vec::new();
    font.set_text(&"= 79.83 ± 0.42 px".chars().map(u32::from).collect::<Vec<_>>(), 0.0, FORCE_AUTOHINT, &mut xys,
        &mut no_warning).unwrap();
    assert_eq!(&xys[..6], &[0.0, 0.0, 1696.0, 0.0, 2408.0, 0.0]);
    assert_eq!((font.get_width_height(), font.get_descent(), font.get_bitmap_offset()), ((20464, 2304), 512, (0, 0)));
    font.draw_glyphs_to_bitmap(true).unwrap();
    let image = font.image();
    assert_eq!((image.height(), image.width()), (38, 320));
    assert_eq!(image.pixels().iter().map(|&value| u64::from(value)).sum::<u64>(), 416295);
}

#[test]
fn empty_string_is_two_by_two() {
    let mut font = font();
    font.set_text(&[], 0.0, FORCE_AUTOHINT, &mut Vec::new(), &mut no_warning).unwrap();
    font.draw_glyphs_to_bitmap(true).unwrap();
    assert_eq!((font.image().width(), font.image().height(), font.image().pixels()), (2, 2, &[0u8; 4][..]));
}

#[test]
fn image_keeps_cpp_clipping() {
    let mut image = FT2Image::new(0, -3).unwrap();
    assert_eq!((image.width(), image.height()), (1, 1));
    image = FT2Image::new(4, 3).unwrap();
    image.draw_rect_filled(2, 1, 10, 1);
    image.draw_rect_filled(0, 0, u32::MAX as _, 0);
    image.draw_rect_filled(u32::MAX as _, 0, 0, 0);
    assert_eq!(image.pixels(), &[0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0]);
}
