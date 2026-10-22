//! Fixed preset geometry: the nine panel presets resolved into normalized axes
//! boxes, the room past each edge, facet packing and typography. Arithmetic
//! follows the scientific layout operation for operation, so every box is the
//! same binary float.

use crate::style::FontStyle;

pub const PANEL_PRESETS: [(&str, u32, u32); 9] = [
    ("1x2", 1, 2), ("2x2", 2, 2), ("4x2", 4, 2), ("1x4", 1, 4), ("2x4", 2, 4),
    ("4x4", 4, 4), ("4x8", 4, 8), ("8x4", 8, 4), ("8x8", 8, 8),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind { Curve, Image, Histogram, Rolling, FacetGrid, PulseTimeline }

impl Kind {
    pub fn parse(name: &str) -> Result<Self, String> {
        let key = name.trim().to_lowercase().replace(['-', ' '], "_");
        Ok(match key.as_str() {
            "curve" => Kind::Curve, "image" => Kind::Image, "histogram" => Kind::Histogram,
            "rolling" => Kind::Rolling, "facet_grid" => Kind::FacetGrid, "pulse_timeline" => Kind::PulseTimeline,
            _ => return Err(format!("unknown plot kind {name:?}")),
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Curve => "curve", Kind::Image => "image", Kind::Histogram => "histogram",
            Kind::Rolling => "rolling", Kind::FacetGrid => "facet_grid", Kind::PulseTimeline => "pulse_timeline",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Margins { pub left: u32, pub right: u32, pub bottom: u32, pub top: u32 }

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageSplit {
    pub image: f64, pub distribution: f64, pub colorbar: f64,
    pub image_distribution_gap: f64, pub distribution_colorbar_gap: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RollingSplit { pub history: f64, pub gap: f64, pub distribution: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutConfig {
    pub default_preset: &'static str,
    pub design_dpi: f64,
    pub display_scale: f64,
    pub export_dpi: f64,
    pub panel_unit: (u32, u32),
    pub panel_margins: Margins,
    pub pulse_left_margin_px: u32,
    pub image_split: ImageSplit,
    pub rolling_split: RollingSplit,
    pub facet_max_columns: u32,
    pub facet_double_rows_threshold: u32,
    pub facet_double_columns_threshold: u32,
    pub facet_column_gap_px: u32,
    pub facet_min_column_gap_px: u32,
    pub facet_row_gap_extra_px: u32,
    pub pulse_row_min_px: u32,
    pub pulse_period_min_px: u32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            default_preset: "2x2", design_dpi: 300.0, display_scale: 0.7, export_dpi: 600.0,
            panel_unit: (240, 180), panel_margins: Margins { left: 110, right: 110, bottom: 80, top: 70 },
            pulse_left_margin_px: 122,
            image_split: ImageSplit { image: 0.75, distribution: 0.10, colorbar: 0.10,
                                      image_distribution_gap: 0.025, distribution_colorbar_gap: 0.025 },
            rolling_split: RollingSplit { history: 0.825, gap: 0.025, distribution: 0.15 },
            facet_max_columns: 7, facet_double_rows_threshold: 4, facet_double_columns_threshold: 5,
            facet_column_gap_px: 10, facet_min_column_gap_px: 6, facet_row_gap_extra_px: 4,
            pulse_row_min_px: 26, pulse_period_min_px: 46,
        }
    }
}

impl LayoutConfig {
    pub fn live_dpi(&self) -> f64 { self.design_dpi * self.display_scale }
    pub fn facet_max_cells(&self) -> u32 { PANEL_PRESETS.iter().map(|&(_, r, c)| r * c).max().unwrap_or(1) }
    pub fn validate_preset(&self, preset: &str) -> Result<&'static str, String> {
        let canonical = preset.trim().to_lowercase().replace(' ', "");
        PANEL_PRESETS.iter().find(|item| item.0 == canonical).map(|item| item.0)
            .ok_or_else(|| format!("unknown panel preset {preset:?}"))
    }
    fn preset(&self, name: &str) -> Result<(&'static str, u32, u32), String> {
        let canonical = self.validate_preset(name)?;
        Ok(*PANEL_PRESETS.iter().find(|item| item.0 == canonical).unwrap())
    }
}

fn round_pixel(value: f64) -> u32 { (value + 0.5).floor().max(1.0) as u32 }

/// Top-origin normalized rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedBox { pub left: f64, pub top: f64, pub right: f64, pub bottom: f64 }

impl NormalizedBox {
    pub fn new(left: f64, top: f64, right: f64, bottom: f64) -> Result<Self, String> {
        if ![left, top, right, bottom].iter().all(|v| v.is_finite()) {
            return Err("box edges must be finite".into());
        }
        if !(0.0 <= left && left < right && right <= 1.0) { return Err("box requires 0 <= left < right <= 1".into()); }
        if !(0.0 <= top && top < bottom && bottom <= 1.0) { return Err("box requires 0 <= top < bottom <= 1".into()); }
        Ok(Self { left, top, right, bottom })
    }
    pub fn width(&self) -> f64 { self.right - self.left }
    pub fn height(&self) -> f64 { self.bottom - self.top }
    /// (left, bottom, width, height) in a bottom-origin figure frame.
    pub fn matplotlib_bounds(&self) -> [f64; 4] { [self.left, 1.0 - self.bottom, self.width(), self.height()] }
}

/// How far past each edge of its box an axes' labels may reach, in figure fractions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Room { pub left: f64, pub right: f64, pub top: f64, pub bottom: f64 }

impl Room {
    pub fn new(left: f64, right: f64, top: f64, bottom: f64) -> Result<Self, String> {
        for value in [left, right, top, bottom] {
            if !value.is_finite() || value < 0.0 { return Err("room must be finite and non-negative".into()); }
        }
        Ok(Self { left, right, top, bottom })
    }
}

fn margin_room(b: &NormalizedBox) -> Result<Room, String> { Room::new(b.left, 1.0 - b.right, b.top, 1.0 - b.bottom) }

#[derive(Clone, Debug, PartialEq)]
pub struct AxesPlan { pub role: String, pub r#box: NormalizedBox, pub cell_index: Option<u32>, pub room: Room }

impl AxesPlan {
    pub fn new(role: &str, r#box: NormalizedBox, cell_index: Option<u32>, room: Room) -> Result<Self, String> {
        let b = &r#box;
        if b.left - room.left < -1e-9 || b.right + room.right > 1.0 + 1e-9
            || b.top - room.top < -1e-9 || b.bottom + room.bottom > 1.0 + 1e-9 {
            return Err("an axes' room cannot leave the figure".into());
        }
        Ok(Self { role: role.trim().to_owned(), r#box, cell_index, room })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FacetTopology { pub cell_count: u32, pub cell_height_over_width: Option<f64> }

#[derive(Clone, Debug, PartialEq)]
pub struct FacetTypography {
    pub tier: &'static str, pub scale: f64, pub cell_title_pt: f64, pub tick_pt: f64,
    pub outer_axis_label_pt: f64, pub cell_title_min_pt: f64, pub cell_title_max_width_pt: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfacePlan {
    pub preset: &'static str,
    pub kind: Kind,
    pub logical_size: (u32, u32),
    pub raster_size: (u32, u32),
    pub figure_size_inches: (f64, f64),
    pub logical_dpi: f64,
    pub dpi: f64,
    pub device_pixel_ratio: f64,
    pub export_scale: f64,
    pub axes: Vec<AxesPlan>,
    pub facet_topology: Option<FacetTopology>,
    pub facet_shape: Option<(u32, u32)>,
    pub facet_typography: Option<FacetTypography>,
    pub facet_focus_axes: Option<Vec<AxesPlan>>,
    pub rolling_side_distribution: bool,
    pub image_height_over_width: Option<f64>,
}

fn panel_design_geometry(preset: (&str, u32, u32), kind: Kind, layout: &LayoutConfig) -> (u32, u32, Margins) {
    let mut margins = layout.panel_margins;
    if kind == Kind::PulseTimeline { margins.left = layout.pulse_left_margin_px; }
    (preset.2 * layout.panel_unit.0, preset.1 * layout.panel_unit.1, margins)
}

fn data_box(data_width: u32, data_height: u32, m: Margins) -> Result<NormalizedBox, String> {
    let width = (m.left + data_width + m.right) as f64;
    let height = (m.top + data_height + m.bottom) as f64;
    NormalizedBox::new(m.left as f64 / width, m.top as f64 / height,
                       (m.left + data_width) as f64 / width, (m.top + data_height) as f64 / height)
}

pub fn facet_shape(cell_count: u32, max_columns: u32) -> (u32, u32) {
    let columns = max_columns.min((cell_count as f64).sqrt().ceil() as u32);
    (((cell_count as f64) / (columns as f64)).ceil() as u32, columns)
}

pub fn facet_shape_for_cell_shape(cell_count: u32, height_over_width: f64, region_px: (f64, f64), max_columns: u32) -> (u32, u32) {
    let cap = max_columns.min(cell_count);
    let mut best: Option<(f64, i64, i64)> = None;
    let mut chosen = (1, cell_count);
    for columns in 1..=cap {
        let rows = ((cell_count as f64) / (columns as f64)).ceil() as u32;
        let cell_width = region_px.0 / columns as f64;
        let cell_height = region_px.1 / rows as f64;
        let image_height = cell_height.min(cell_width * height_over_width);
        let key = (image_height, -((rows * columns) as i64), -((rows as i64 - columns as i64).abs()));
        if best.is_none_or(|current| key > current) { best = Some(key); chosen = (rows, columns); }
    }
    chosen
}

fn facet_cells_union(axes: &[AxesPlan]) -> Result<NormalizedBox, String> {
    let boxes: Vec<_> = axes.iter().filter(|a| a.role == "facet_cell").map(|a| a.r#box).collect();
    if boxes.is_empty() { return Err("FacetGrid surface has no cell geometry".into()); }
    let min = |f: fn(&NormalizedBox) -> f64| boxes.iter().map(f).fold(f64::INFINITY, f64::min);
    let max = |f: fn(&NormalizedBox) -> f64| boxes.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
    NormalizedBox::new(min(|b| b.left), min(|b| b.top), max(|b| b.right), max(|b| b.bottom))
}

pub fn facet_focus_box(plan: &SurfacePlan) -> Result<NormalizedBox, String> {
    if plan.kind != Kind::FacetGrid { return Err("facet focus geometry requires a FacetGrid surface".into()); }
    facet_cells_union(&plan.axes)
}

pub fn recommended_facet_preset(topology: FacetTopology, layout: &LayoutConfig) -> Result<&'static str, String> {
    let (rows, columns) = facet_shape(topology.cell_count, layout.facet_max_columns);
    let row_units = if rows > layout.facet_double_rows_threshold { 4 } else { 2 };
    let column_units = if columns > layout.facet_double_columns_threshold { 4 } else { 2 };
    layout.validate_preset(&format!("{row_units}x{column_units}"))
}

fn facet_scale(selected: (&str, u32, u32), recommended: (&str, u32, u32), fonts: &FontStyle) -> (&'static str, f64) {
    let compact = selected.1 < recommended.1 || selected.2 < recommended.2;
    if compact { ("compact", fonts.facet_compact_scale) } else { ("normal", fonts.facet_normal_scale) }
}

fn facet_gaps(scale: f64, fonts: &FontStyle, layout: &LayoutConfig) -> (u32, u32) {
    let row_gap = round_pixel(fonts.tick_pt * scale * layout.design_dpi / 72.0) + layout.facet_row_gap_extra_px;
    let column_gap = layout.facet_min_column_gap_px.max(round_pixel(layout.facet_column_gap_px as f64 * scale));
    (row_gap, column_gap)
}

fn split_image(data: &NormalizedBox, split: &ImageSplit, region_px: Option<(f64, f64)>,
               image_height_over_width: Option<f64>, scene: bool, side_distribution: bool) -> Result<Vec<AxesPlan>, String> {
    let mut span = data.width() * split.image;
    let mut unit = data.width();
    if let (Some(shape), Some((width_px, height_px))) = (image_height_over_width, region_px) {
        if width_px > 0.0 && height_px > 0.0 && shape > 0.0 {
            let drawn_px = (width_px * split.image).min(height_px / shape);
            let drawn = data.width() * (drawn_px / width_px);
            if drawn < span { span = drawn; unit = span / split.image; }
        }
    }
    let cursor = data.left;
    let mut image = NormalizedBox::new(cursor, data.top, cursor + span, data.bottom)?;
    let cursor = image.right + unit * split.image_distribution_gap;
    let distribution = NormalizedBox::new(cursor, data.top, cursor + unit * split.distribution, data.bottom)?;
    let cursor = distribution.right + unit * split.distribution_colorbar_gap;
    let colorbar = NormalizedBox::new(cursor, data.top, cursor + unit * split.colorbar, data.bottom)?;
    if scene { image = scene_box(&image, data, split, region_px, &distribution)?; }
    if !side_distribution { return Ok(vec![AxesPlan::new("image", image, None, margin_room(&image)?)?]); }
    let rail_gap = (distribution.left - image.right).max(0.0) / 2.0;
    let bar_gap = (colorbar.left - distribution.right).max(0.0) / 2.0;
    Ok(vec![
        AxesPlan::new("image", image, None, Room::new(image.left, rail_gap, image.top, 1.0 - image.bottom)?)?,
        AxesPlan::new("distribution", distribution, None,
                      Room::new(rail_gap, bar_gap, distribution.top, 1.0 - distribution.bottom)?)?,
        AxesPlan::new("colorbar", colorbar, None,
                      Room::new(bar_gap, 1.0 - colorbar.right, colorbar.top, 1.0 - colorbar.bottom)?)?,
    ])
}

fn scene_box(image: &NormalizedBox, data: &NormalizedBox, split: &ImageSplit, region_px: Option<(f64, f64)>,
             distribution: &NormalizedBox) -> Result<NormalizedBox, String> {
    let pad_x = split.image_distribution_gap * (image.width() / split.image);
    let mut pad_y = pad_x;
    if let Some((width_px, height_px)) = region_px {
        if width_px > 0.0 && height_px > 0.0 {
            let figure_w = width_px / data.width();
            let figure_h = height_px / data.height();
            if figure_h > 0.0 { pad_y = pad_x * figure_w / figure_h; }
        }
    }
    let left = pad_x.min(image.left);
    let right = (distribution.left - pad_x).max(image.right);
    let bottom = (1.0 - pad_y).max(image.bottom);
    NormalizedBox::new(left, image.top, right.min(1.0), bottom.min(1.0))
}

fn split_rolling(data: &NormalizedBox, split: &RollingSplit, show_distribution: bool) -> Result<Vec<AxesPlan>, String> {
    if !show_distribution { return Ok(vec![AxesPlan::new("history", *data, None, margin_room(data)?)?]); }
    let history = NormalizedBox::new(data.left, data.top, data.left + data.width() * split.history, data.bottom)?;
    let distribution = NormalizedBox::new(history.right + data.width() * split.gap, data.top, data.right, data.bottom)?;
    let half_gap = (distribution.left - history.right).max(0.0) / 2.0;
    Ok(vec![
        AxesPlan::new("history", history, None, Room::new(history.left, half_gap, history.top, 1.0 - history.bottom)?)?,
        AxesPlan::new("distribution", distribution, None,
                      Room::new(half_gap, 1.0 - distribution.right, distribution.top, 1.0 - distribution.bottom)?)?,
    ])
}

fn facet_axes(count: u32, shape: (u32, u32), data_width: u32, data_height: u32, m: Margins,
              row_gap: u32, column_gap: u32) -> Result<Vec<AxesPlan>, String> {
    let (rows, columns) = shape;
    let cell_width = (((data_width as i64 - (columns as i64 - 1) * column_gap as i64) as f64) / columns as f64).max(1.0);
    let cell_height = (((data_height as i64 - (rows as i64 - 1) * row_gap as i64) as f64) / rows as f64).max(1.0);
    let figure_width = (m.left + data_width + m.right) as f64;
    let figure_height = (m.bottom + data_height + m.top) as f64;
    let room = Room::new(column_gap as f64 / 2.0 / figure_width, column_gap as f64 / 2.0 / figure_width,
                         row_gap as f64 / 2.0 / figure_height, row_gap as f64 / 2.0 / figure_height)?;
    (0..count).map(|index| {
        let (row, column) = (index / columns, index % columns);
        let left = m.left as f64 + column as f64 * (cell_width + column_gap as f64);
        let bottom_from_origin = m.bottom as f64 + (rows - 1 - row) as f64 * (cell_height + row_gap as f64);
        let top = figure_height - (bottom_from_origin + cell_height);
        AxesPlan::new("facet_cell", NormalizedBox::new(left / figure_width, top / figure_height,
            (left + cell_width) / figure_width, (top + cell_height) / figure_height)?, Some(index), room)
    }).collect()
}

/// Options of one surface resolution beside its preset and kind.
#[derive(Clone, Debug, Default)]
pub struct SurfaceOptions {
    pub facet_topology: Option<FacetTopology>,
    pub device_pixel_ratio: Option<f64>,
    pub export_scale: Option<f64>,
    pub rolling_side_distribution: Option<bool>,
    pub image_height_over_width: Option<f64>,
    pub image_scene: bool,
    pub image_side_distribution: Option<bool>,
}

pub fn resolve_surface(preset: Option<&str>, kind: Kind, options: &SurfaceOptions, layout: &LayoutConfig,
                       fonts: &FontStyle) -> Result<SurfacePlan, String> {
    if kind == Kind::Rolling {
        if options.rolling_side_distribution.is_none() {
            return Err("Rolling surfaces require an explicit rolling_side_distribution".into());
        }
    } else if options.rolling_side_distribution.is_some() {
        return Err("rolling_side_distribution is accepted only for Rolling surfaces".into());
    }
    if options.image_height_over_width.is_some() && kind != Kind::Image {
        return Err("image_height_over_width is accepted only for Image surfaces".into());
    }
    if options.image_scene && !matches!(kind, Kind::Image | Kind::FacetGrid) {
        return Err("image_scene is accepted only for Image and FacetGrid surfaces".into());
    }
    let side_distribution = options.image_side_distribution.unwrap_or(true);
    let (selected_name, recommended_name) = if kind == Kind::FacetGrid {
        let topology = options.facet_topology.ok_or("FacetGrid requires facet_topology")?;
        if topology.cell_count > layout.facet_max_cells() {
            return Err(format!("FacetGrid needs {} cells, which exceeds the fixed layout facet_max_cells capacity of {}; \
                                pin an axis or change a fate to reduce the cells", topology.cell_count, layout.facet_max_cells()));
        }
        let recommended = recommended_facet_preset(topology, layout)?;
        (match preset { None => recommended, Some(p) => layout.validate_preset(p)? }, Some(recommended))
    } else {
        if options.facet_topology.is_some() { return Err("facet_topology is accepted only for FacetGrid".into()); }
        (match preset { None => layout.default_preset, Some(p) => layout.validate_preset(p)? }, None)
    };
    let selected = layout.preset(selected_name)?;
    let (data_width, data_height, margins) = panel_design_geometry(selected, kind, layout);
    let figure_design_width = margins.left + data_width + margins.right;
    let figure_design_height = margins.top + data_height + margins.bottom;
    let logical_size = (round_pixel(figure_design_width as f64 * layout.display_scale),
                        round_pixel(figure_design_height as f64 * layout.display_scale));
    let dpr = options.device_pixel_ratio.unwrap_or(1.0);
    if !(dpr.is_finite() && dpr > 0.0) { return Err("device_pixel_ratio must be positive and finite".into()); }
    let physical_export_scale = options.export_scale.unwrap_or(1.0);
    if !(physical_export_scale.is_finite() && physical_export_scale > 0.0) {
        return Err("export_scale must be positive and finite".into());
    }
    let physical_scale = dpr * physical_export_scale;
    let physical_dpi = layout.live_dpi() * physical_scale;
    let raster_size = (round_pixel(logical_size.0 as f64 * physical_scale), round_pixel(logical_size.1 as f64 * physical_scale));
    let figure_size_inches = (raster_size.0 as f64 / physical_dpi, raster_size.1 as f64 / physical_dpi);
    let data = data_box(data_width, data_height, margins)?;
    let (mut typography, mut shape, mut show_distribution, mut facet_focus_axes) = (None, None, false, None);
    let axes = match kind {
        Kind::Image => split_image(&data, &layout.image_split, Some((data_width as f64, data_height as f64)),
                                   options.image_height_over_width, options.image_scene, side_distribution)?,
        Kind::Rolling => {
            show_distribution = options.rolling_side_distribution.unwrap();
            split_rolling(&data, &layout.rolling_split, show_distribution)?
        }
        Kind::FacetGrid => {
            let topology = options.facet_topology.unwrap();
            let recommended = layout.preset(recommended_name.unwrap())?;
            let (tier, scale) = facet_scale(selected, recommended, fonts);
            let (row_gap, column_gap) = facet_gaps(scale, fonts, layout);
            let packed = match topology.cell_height_over_width {
                None => facet_shape(topology.cell_count, layout.facet_max_columns),
                Some(aspect) => facet_shape_for_cell_shape(topology.cell_count, aspect,
                    (data_width as f64, data_height as f64), layout.facet_max_columns),
            };
            let columns = packed.1;
            let cell_width = (((data_width as i64 - (columns as i64 - 1) * column_gap as i64) as f64) / columns as f64).max(1.0);
            let cell_title_pt = fonts.tick_pt * scale;
            typography = Some(FacetTypography {
                tier, scale, cell_title_pt, tick_pt: fonts.tick_pt * scale,
                outer_axis_label_pt: fonts.axis_label_pt,
                cell_title_min_pt: fonts.facet_title_min_pt.min(cell_title_pt),
                cell_title_max_width_pt: (cell_width + column_gap as f64) * 72.0 / layout.design_dpi,
            });
            let cells = facet_axes(topology.cell_count, packed, data_width, data_height, margins, row_gap, column_gap)?;
            let union = facet_cells_union(&cells)?;
            facet_focus_axes = Some(split_image(&union, &layout.image_split,
                Some((union.width() * figure_design_width as f64, union.height() * figure_design_height as f64)),
                topology.cell_height_over_width, options.image_scene, side_distribution)?);
            shape = Some(packed);
            cells
        }
        _ => vec![AxesPlan::new("main", data, None, margin_room(&data)?)?],
    };
    Ok(SurfacePlan {
        preset: selected.0, kind, logical_size, raster_size, figure_size_inches,
        logical_dpi: layout.live_dpi(), dpi: physical_dpi, device_pixel_ratio: dpr,
        export_scale: physical_export_scale, axes, facet_topology: options.facet_topology,
        facet_shape: shape, facet_typography: typography, facet_focus_axes,
        rolling_side_distribution: show_distribution,
        image_height_over_width: match options.facet_topology {
            None => options.image_height_over_width,
            Some(topology) => topology.cell_height_over_width,
        },
    })
}
