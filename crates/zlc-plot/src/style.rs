//! The plot style: the scientific style's immutable design tokens, one field
//! per configured value (`zlc_plot.style.build_plot_style`). Colours stay the
//! Matplotlib colour specifications the tokens are written in; resolving
//! them to RGBA is the drawing layer's job.

#[derive(Clone, Debug, PartialEq)]
pub struct FontStyle {
    pub family: &'static str,
    pub fallbacks: &'static [&'static str],
    pub weight: &'static str,
    pub figure_title_pt: f64,
    pub axis_label_pt: f64,
    pub tick_pt: f64,
    pub legend_pt: f64,
    pub annotation_pt: f64,
    pub fit_annotation_pt: f64,
    pub facet_fit_annotation_pt: f64,
    pub facet_normal_scale: f64,
    pub facet_compact_scale: f64,
    pub facet_title_min_pt: f64,
    pub pulse_repeat_pt: f64,
    pub pulse_bar_label_pt: f64,
    pub pulse_scan_annotation_pt: f64,
}

impl Default for FontStyle {
    fn default() -> Self {
        Self {
            family: "Helvetica Light", fallbacks: &["Arial"], weight: "normal",
            figure_title_pt: 7.5, axis_label_pt: 7.5, tick_pt: 6.5, legend_pt: 6.5, annotation_pt: 6.5,
            fit_annotation_pt: 3.25, facet_fit_annotation_pt: 3.5, facet_normal_scale: 1.0,
            facet_compact_scale: 0.5, facet_title_min_pt: 3.25, pulse_repeat_pt: 5.7,
            pulse_bar_label_pt: 5.3, pulse_scan_annotation_pt: 3.36,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Palette {
    pub series: &'static [&'static str],
    pub line_single: &'static str,
    pub pulse_cycle: &'static [&'static str],
    pub bracket_cycle: &'static [&'static str],
    pub pulse_run: &'static str,
    pub hist_fill: &'static str,
    pub bright: &'static str,
    pub fit_left: &'static str,
    pub fit_right: &'static str,
    pub fit_total: &'static str,
    pub threshold: &'static str,
    pub annotation: &'static str,
    pub readout: &'static str,
    pub fit_text: &'static str,
    pub guide: &'static str,
    pub data_scatter: &'static str,
    pub pulse_name: &'static str,
    pub pulse_grid: &'static str,
    pub pulse_period: &'static str,
}

impl Palette {
    /// Shared series inks; group zero and ungrouped traces use grey.
    pub fn line_cycle(&self) -> Vec<&'static str> {
        std::iter::once(self.line_single).chain(self.series.iter().copied()).collect()
    }

    /// The canonical colour for a zero-based source-series index.
    pub fn line_color(&self, index: usize) -> &'static str {
        if index == 0 { return self.line_single; }
        let cycle = self.series.len() + 1;
        let position = index % cycle;
        if position == 0 { self.line_single } else { self.series[position - 1] }
    }

    /// The ink of the bracket numbered `index` (0 = outermost).
    pub fn bracket_color(&self, index: usize) -> &'static str { self.bracket_cycle[index % self.bracket_cycle.len()] }
}

impl Default for Palette {
    fn default() -> Self {
        Palette {
            series: &["#517FA6", "#B17D4B", "#568C79", "#99719B", "#B36F72", "#558F9C", "#7E7CA5", "#92924F"],
            line_single: "#808080",
            pulse_cycle: &["#5D7583", "#C37D5A", "#6F8D73", "#A66E87", "#7A6FA4", "#B5A262", "#5E9A9A", "#9A765E",
                           "#7890B5", "#8B8B8B", "#B97878", "#679174"],
            bracket_cycle: &["#C96F3D", "#4F7EA8", "#8B6BB8", "#5B9A6B"],
            pulse_run: "#6A6866", hist_fill: "grey", bright: "skyblue", fit_left: "skyblue", fit_right: "orange",
            fit_total: "black", threshold: "orange", annotation: "black", readout: "grey", fit_text: "blue",
            guide: "grey", data_scatter: "lightgrey", pulse_name: "white", pulse_grid: "0.88", pulse_period: "#4A4A4A",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LineToken {
    pub color: &'static str,
    pub linewidth: f64,
    pub alpha: f64,
    pub linestyle: &'static str,
    pub marker: Option<&'static str>,
    pub zorder: Option<f64>,
}

impl LineToken {
    pub const fn new(color: &'static str, linewidth: f64, alpha: f64) -> Self {
        LineToken { color, linewidth, alpha, linestyle: "-", marker: None, zorder: None }
    }
    pub const fn at(mut self, zorder: f64) -> Self { self.zorder = Some(zorder); self }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointRingToken { pub color: &'static str, pub alpha: f64, pub linewidth: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct ArtistStyle {
    pub curve: LineToken,
    pub curve_marker_size_pt: f64,
    pub histogram_fill_alpha: f64,
    pub threshold_line: LineToken,
    pub classifier_threshold_line: LineToken,
    pub curve_fit_line: LineToken,
    pub bimodal_fit_lines: [LineToken; 3],
    pub fit_failure_color: &'static str,
    pub fit_ellipse_color: &'static str,
    pub fit_ellipse_center_area_pt2: f64,
    pub side_distribution_linewidth: f64,
    pub pulse_scan_region_color: &'static str,
    pub pulse_api_region_color: &'static str,
    pub pulse_scan_annotation_color: &'static str,
    pub point_unknown: PointRingToken,
    pub point_empty: PointRingToken,
    pub point_occupied: PointRingToken,
    pub point_invalid: PointRingToken,
    pub point_auto_radius_fraction: f64,
    pub point_single_radius_fraction: f64,
    pub point_zorder: f64,
    pub point_label_zorder: f64,
    pub selector_line_width: f64,
    pub selector_alpha: f64,
    pub selector_handle_size_pt: f64,
    pub selector_handle_edge_width: f64,
    pub selector_zorder: f64,
    pub fit_annotation_zorder: f64,
    pub crosshair_marker_size_pt: f64,
    pub fit_source_scatter_area_pt2: f64,
    pub fit_source_scatter_zorder: f64,
    pub fit_ellipse_zorder: f64,
    pub fit_component_sample_count: usize,
}

impl ArtistStyle {
    fn from_palette(palette: &Palette) -> Self {
        ArtistStyle {
            curve: LineToken::new(palette.line_single, 1.0, 0.8),
            curve_marker_size_pt: 6.0,
            histogram_fill_alpha: 0.4,
            threshold_line: LineToken::new(palette.threshold, 1.9, 0.95).at(5.0),
            classifier_threshold_line: LineToken::new(palette.threshold, 0.9, 0.95).at(5.0),
            curve_fit_line: LineToken::new(palette.fit_right, 2.0, 0.5).at(5.0),
            bimodal_fit_lines: [LineToken::new(palette.fit_left, 1.0, 0.8), LineToken::new(palette.fit_right, 1.0, 0.8),
                                LineToken::new(palette.fit_total, 1.0, 0.35)],
            fit_failure_color: "#CD7380",
            fit_ellipse_color: palette.fit_right,
            fit_ellipse_center_area_pt2: 2.25,
            side_distribution_linewidth: 3.25,
            pulse_scan_region_color: "#D69A6E",
            pulse_api_region_color: "#B08BD6",
            pulse_scan_annotation_color: "#FFFFFF",
            point_unknown: PointRingToken { color: "#FFFFFF", alpha: 0.45, linewidth: 0.6 },
            point_empty: PointRingToken { color: "#FFFFFF", alpha: 0.12, linewidth: 0.5 },
            point_occupied: PointRingToken { color: "#D07850", alpha: 0.58, linewidth: 0.85 },
            point_invalid: PointRingToken { color: "#CD7380", alpha: 0.50, linewidth: 0.7 },
            point_auto_radius_fraction: 0.3,
            point_single_radius_fraction: 0.03,
            point_zorder: 5.0,
            point_label_zorder: 6.0,
            selector_line_width: 1.0,
            selector_alpha: 0.8,
            selector_handle_size_pt: 3.25,
            selector_handle_edge_width: 0.5,
            selector_zorder: 20.0,
            fit_annotation_zorder: 15.0,
            crosshair_marker_size_pt: 2.0,
            fit_source_scatter_area_pt2: 20.0,
            fit_source_scatter_zorder: 4.0,
            fit_ellipse_zorder: 6.0,
            fit_component_sample_count: 300,
        }
    }
}

/// A dash pattern: offset and on/off lengths, in points.
pub type Dash = (f64, &'static [f64]);

#[derive(Clone, Debug, PartialEq)]
pub struct PulseStyle {
    pub row_height: f64,
    pub dense_row_threshold: usize,
    pub dense_min_row_height: f64,
    pub dense_total_height: f64,
    pub x_margin_fraction: f64,
    pub grid_linewidth: f64,
    pub trace_linewidth: f64,
    pub label_fit_pad_pt: f64,
    pub period_band_height: f64,
    pub period_boundary_linewidth: f64,
    pub period_boundary_alpha: f64,
    pub period_boundary_dash: Dash,
    pub spacer_hatch: &'static str,
    pub spacer_alpha: f64,
    pub analog_zero_alpha: f64,
    pub analog_zero_dash: Dash,
    pub scan_region_alpha: f64,
    pub scan_badge_pad: f64,
    pub scan_dac_linewidth: f64,
    pub scan_dac_alpha: f64,
    pub repeat_foot_axes_fraction: f64,
    pub repeat_max_foot_fraction: f64,
    pub repeat_bottom: f64,
    pub repeat_bottom_step: f64,
    pub repeat_top_offset: f64,
    pub repeat_top_step_pt: f64,
    pub repeat_ylim_room_pt: f64,
    pub repeat_alpha: f64,
    pub repeat_linewidth: f64,
    pub repeat_label_offset_pt: (f64, f64),
    pub ylim_bottom: f64,
    pub ylim_top_offset: f64,
    pub ytick_font_floor_pt: f64,
    pub ytick_font_delta_pt: f64,
    pub xtick_count: usize,
    pub xtick_pad_pt: f64,
    pub base_zorder: f64,
    pub scan_region_zorder: f64,
    pub scan_dac_zorder: f64,
    pub annotation_zorder: f64,
    pub repeat_bracket_zorder: f64,
    pub repeat_label_zorder: f64,
}

impl Default for PulseStyle {
    fn default() -> Self {
        PulseStyle {
            row_height: 0.64, dense_row_threshold: 10, dense_min_row_height: 0.42, dense_total_height: 6.4,
            x_margin_fraction: 0.04, grid_linewidth: 0.35, trace_linewidth: 0.65, label_fit_pad_pt: 1.5,
            period_band_height: 0.6, period_boundary_linewidth: 0.8, period_boundary_alpha: 0.45,
            period_boundary_dash: (0.0, &[3.0, 2.2]), spacer_hatch: "////", spacer_alpha: 0.22,
            analog_zero_alpha: 0.5, analog_zero_dash: (0.0, &[4.0, 3.0]), scan_region_alpha: 0.18,
            scan_badge_pad: 0.3, scan_dac_linewidth: 3.0, scan_dac_alpha: 0.9, repeat_foot_axes_fraction: 0.012,
            repeat_max_foot_fraction: 0.2, repeat_bottom: -0.42, repeat_bottom_step: 0.13, repeat_top_offset: -0.10,
            repeat_top_step_pt: 9.0, repeat_ylim_room_pt: 11.0, repeat_alpha: 0.58, repeat_linewidth: 1.05,
            repeat_label_offset_pt: (-2.0, 1.5), ylim_bottom: -0.62, ylim_top_offset: -0.38,
            ytick_font_floor_pt: 4.8, ytick_font_delta_pt: 1.2, xtick_count: 5, xtick_pad_pt: 2.0,
            base_zorder: 3.0, scan_region_zorder: 6.0, scan_dac_zorder: 8.0, annotation_zorder: 12.0,
            repeat_bracket_zorder: 8.0, repeat_label_zorder: 9.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderPolicy {
    pub image_default_colormap: &'static str,
    pub image_colormaps: &'static [&'static str],
    /// `ImageFrontPolicy.minimum_reduction_ratio`.
    pub image_front_minimum_reduction_ratio: f64,
    pub image_origin: &'static str,
    pub image_anchor: &'static str,
    pub height_bars_side_shades: (f64, f64),
    pub height_bars_background_rgb: (f64, f64, f64),
    pub height_bars_axis_color: &'static str,
    pub height_bars_grid_rgb: (f64, f64, f64),
    pub height_bars_grid_line_pt: f64,
    pub height_bars_grid_alpha: f64,
    pub height_bars_z_fraction: f64,
    pub height_bars_cage_alpha: f64,
    pub axes_title_pad_pt: f64,
    pub compact_axes_title_pad_pt: f64,
    pub facet_cell_tick_length_pt: f64,
    pub figure_title_y: f64,
    pub colorbar_limit_label_chars: usize,
    pub axes_text_inset_fraction: f64,
    pub selector_label_line_fraction: f64,
    pub selector_sample_colormap_fraction: f64,
    pub selector_sample_alpha: f64,
    pub colormap_low_fraction: f64,
    pub colormap_high_fraction: f64,
    pub side_distribution_fill_alpha: f64,
    pub uncertainty_bar_alpha: f64,
    pub uncertainty_bar_linewidth: f64,
    pub uncertainty_bar_capsize_pt: f64,
    pub rolling_distribution_min_bins: usize,
    pub image_distribution_min_bins: usize,
    pub distribution_max_bins: usize,
    pub distribution_bin_divisor: usize,
    pub distribution_guide_alpha: f64,
    pub distribution_tick_count: usize,
    /// The image rail's count labels hang inside it from its ends (v2 centred
    /// them on the ends, so the peak count ran under the colorbar).
    pub image_distribution_counts_inside: bool,
    pub image_distribution_sample_target: usize,
    pub image_color_padding_fraction: f64,
    pub image_color_deadband_fraction: f64,
    pub distribution_count_floor: usize,
    pub distribution_count_headroom: f64,
    pub distribution_count_growth: f64,
    pub distribution_count_shrink_ratio: f64,
    pub fit_source_scatter_max_points: usize,
}

impl Default for RenderPolicy {
    fn default() -> Self {
        RenderPolicy {
            image_default_colormap: "gray",
            image_colormaps: &["inferno", "viridis", "magma", "plasma", "gray", "RdBu_r", "Blues"],
            image_front_minimum_reduction_ratio: 1.25,
            image_origin: "upper", image_anchor: "W",
            height_bars_side_shades: (1.0, 1.0), height_bars_background_rgb: (1.0, 1.0, 1.0),
            height_bars_axis_color: "black", height_bars_grid_rgb: (0.45, 0.45, 0.50), height_bars_grid_line_pt: 1.0,
            height_bars_grid_alpha: 0.1, height_bars_z_fraction: 0.55, height_bars_cage_alpha: 0.2,
            axes_title_pad_pt: 2.5, compact_axes_title_pad_pt: 1.5, facet_cell_tick_length_pt: 2.0,
            figure_title_y: 0.992, colorbar_limit_label_chars: 5, axes_text_inset_fraction: 0.025,
            selector_label_line_fraction: 0.11, selector_sample_colormap_fraction: 0.55, selector_sample_alpha: 0.3,
            colormap_low_fraction: 0.0, colormap_high_fraction: 0.95, side_distribution_fill_alpha: 1.0,
            uncertainty_bar_alpha: 0.45, uncertainty_bar_linewidth: 1.1, uncertainty_bar_capsize_pt: 1.2,
            rolling_distribution_min_bins: 3, image_distribution_min_bins: 8, distribution_max_bins: 50,
            distribution_bin_divisor: 4, distribution_guide_alpha: 0.3, distribution_tick_count: 2, image_distribution_counts_inside: true,
            image_distribution_sample_target: 50_000, image_color_padding_fraction: 0.10,
            image_color_deadband_fraction: 0.35, distribution_count_floor: 10, distribution_count_headroom: 5.0,
            distribution_count_growth: 1.5, distribution_count_shrink_ratio: 0.6, fit_source_scatter_max_points: 2000,
        }
    }
}

/// One Matplotlib rcParam the style sets.
#[derive(Clone, Debug, PartialEq)]
pub enum RcValue { Text(&'static str), Number(f64), Integer(i64), Flag(bool), Texts(Vec<&'static str>) }

/// The complete visual style.
#[derive(Clone, Debug, PartialEq)]
pub struct PlotStyle {
    pub fonts: FontStyle,
    pub palette: Palette,
    pub artists: ArtistStyle,
    pub pulse: PulseStyle,
    pub render: RenderPolicy,
    pub rc_params: Vec<(&'static str, RcValue)>,
}

impl Default for PlotStyle {
    fn default() -> Self {
        use RcValue::{Flag, Number, Integer, Text};
        let palette = Palette::default();
        PlotStyle {
            fonts: FontStyle::default(),
            artists: ArtistStyle::from_palette(&palette),
            palette,
            pulse: PulseStyle::default(),
            render: RenderPolicy::default(),
            rc_params: vec![
                ("scatter.edgecolors", Text("black")), ("legend.numpoints", Integer(1)),
                ("ytick.major.size", Number(1.5)), ("ytick.major.width", Number(0.4)),
                ("xtick.major.size", Number(1.5)), ("xtick.major.width", Number(0.4)),
                ("axes.linewidth", Number(0.4)), ("figure.subplot.left", Number(0.0)),
                ("figure.subplot.right", Number(1.0)), ("figure.subplot.bottom", Number(0.0)),
                ("figure.subplot.top", Number(1.0)), ("xtick.major.pad", Number(1.5)),
                ("ytick.major.pad", Number(1.5)), ("axes.labelpad", Number(1.5)), ("grid.linestyle", Text("--")),
                ("axes.grid", Flag(false)), ("axes.facecolor", Text("white")), ("figure.facecolor", Text("white")),
                ("figure.autolayout", Flag(false)), ("figure.constrained_layout.use", Flag(false)),
                ("text.usetex", Flag(false)), ("mathtext.fontset", Text("dejavusans")), ("mathtext.fallback", Text("cm")),
                ("xtick.top", Flag(false)), ("ytick.right", Flag(false)), ("xtick.minor.top", Flag(false)),
                ("ytick.minor.right", Flag(false)), ("xtick.minor.bottom", Flag(false)), ("ytick.minor.left", Flag(false)),
                ("xtick.direction", Text("in")), ("ytick.direction", Text("in")), ("legend.frameon", Flag(false)),
                ("savefig.facecolor", Text("white")), ("savefig.edgecolor", Text("white")),
                ("savefig.transparent", Flag(false)), ("text.color", Text("black")), ("patch.edgecolor", Text("black")),
                ("patch.force_edgecolor", Flag(false)), ("hatch.color", Text("black")), ("axes.edgecolor", Text("black")),
                ("axes.titlecolor", Text("black")), ("axes.labelcolor", Text("black")), ("xtick.color", Text("black")),
                ("ytick.color", Text("black")),
            ],
        }
    }
}

impl PlotStyle {
    /// The rc params the style derives from its own typed policy.
    pub fn derived_rc_params(&self) -> Vec<(&'static str, RcValue)> {
        use RcValue::{Number, Text, Texts};
        let (fonts, curve, render) = (&self.fonts, &self.artists.curve, &self.render);
        let families: Vec<&'static str> = std::iter::once(fonts.family).chain(fonts.fallbacks.iter().copied()).collect();
        vec![
            ("font.size", Number(fonts.annotation_pt)), ("font.weight", Text(fonts.weight)),
            ("font.family", Texts(families.clone())), ("font.sans-serif", Texts(families)),
            ("axes.labelsize", Number(fonts.axis_label_pt)), ("axes.labelweight", Text(fonts.weight)),
            ("axes.titlesize", Number(fonts.figure_title_pt)), ("axes.titleweight", Text(fonts.weight)),
            ("axes.titlepad", Number(render.compact_axes_title_pad_pt)), ("figure.titlesize", Number(fonts.figure_title_pt)),
            ("figure.titleweight", Text(fonts.weight)), ("legend.fontsize", Number(fonts.legend_pt)),
            ("legend.title_fontsize", Number(fonts.legend_pt)), ("xtick.labelsize", Number(fonts.tick_pt)),
            ("ytick.labelsize", Number(fonts.tick_pt)), ("lines.linewidth", Number(curve.linewidth)),
            ("lines.linestyle", Text(curve.linestyle)), ("lines.marker", Text(curve.marker.unwrap_or("None"))),
            ("lines.markersize", Number(self.artists.curve_marker_size_pt)),
            ("image.cmap", Text(render.image_default_colormap)), ("image.origin", Text(render.image_origin)),
        ]
    }
}
