//! The rcParams the renderer's artists are built under: Matplotlib 3.10's
//! defaults (`rcParamsOrig` of the product environment) with the style's
//! own params and the params it derives laid over them (`style_context`).
use zlc_figure::ir;

use crate::style::{PlotStyle, RcValue};

/// One axis family's tick rc (`xtick.*` / `ytick.*`).
#[derive(Clone, Debug, PartialEq)]
pub struct TickRc {
    pub major_size: f64,
    pub major_width: f64,
    pub major_pad: f64,
    pub minor_size: f64,
    pub minor_width: f64,
    pub minor_pad: f64,
    pub color: String,
    pub labelcolor: String,
    pub labelsize: f64,
    pub direction: String,
    /// x: (bottom, top) / y: (left, right) tick marks.
    pub tick1: bool,
    pub tick2: bool,
    /// x: (labelbottom, labeltop) / y: (labelleft, labelright).
    pub label1: bool,
    pub label2: bool,
}

#[derive(Clone, Debug)]
pub struct Rc {
    pub font_family: Vec<String>,
    pub font_size: f64,
    pub font_weight: String,
    pub math_fontfamily: String,
    pub text_color: String,
    pub axes_labelsize: f64,
    /// `axes.xmargin` / `axes.ymargin`.
    pub axes_margins: (f64, f64),
    pub axes_labelpad: f64,
    pub axes_labelcolor: String,
    pub axes_titlesize: f64,
    pub axes_titlepad: f64,
    pub axes_titlecolor: String,
    pub axes_linewidth: f64,
    pub axes_edgecolor: String,
    pub axes_facecolor: String,
    /// `axes.axisbelow` 'line' puts the axis at zorder 1.5.
    pub axis_zorder: f64,
    pub figure_facecolor: String,
    pub figure_edgecolor: String,
    pub grid_color: String,
    pub grid_linestyle: String,
    pub grid_linewidth: f64,
    pub grid_alpha: f64,
    pub lines_linewidth: f64,
    pub lines_markersize: f64,
    pub lines_markeredgewidth: f64,
    pub lines_dashed_pattern: [f64; 2],
    pub lines_dashdot_pattern: [f64; 4],
    pub lines_dotted_pattern: [f64; 2],
    pub patch_linewidth: f64,
    pub xtick: TickRc,
    pub ytick: TickRc,
    pub ir: ir::Rc,
}

impl Rc {
    /// The rc mapping `style_context(style)` resolves.
    pub fn new(style: &PlotStyle) -> Rc {
        let mut rc = Rc::matplotlib_defaults();
        for (name, value) in style.rc_params.iter().chain(style.derived_rc_params().iter()) {
            rc.set(name, value);
        }
        rc
    }

    fn tick_defaults(x: bool) -> TickRc {
        TickRc {
            major_size: 3.5, major_width: 0.8, major_pad: 3.5, minor_size: 2.0, minor_width: 0.6, minor_pad: 3.4,
            color: "black".into(), labelcolor: "inherit".into(), labelsize: 10.0, direction: "out".into(),
            tick1: true, tick2: false, label1: true, label2: false,
        }
        .with_x(x)
    }

    fn matplotlib_defaults() -> Rc {
        Rc {
            font_family: vec!["sans-serif".into()],
            font_size: 10.0,
            font_weight: "normal".into(),
            math_fontfamily: "dejavusans".into(),
            text_color: "black".into(),
            axes_labelsize: 10.0,
            axes_margins: (0.05, 0.05),
            axes_labelpad: 4.0,
            axes_labelcolor: "black".into(),
            axes_titlesize: 12.0,
            axes_titlepad: 6.0,
            axes_titlecolor: "auto".into(),
            axes_linewidth: 0.8,
            axes_edgecolor: "black".into(),
            axes_facecolor: "white".into(),
            axis_zorder: 1.5,
            figure_facecolor: "white".into(),
            figure_edgecolor: "white".into(),
            grid_color: "#b0b0b0".into(),
            grid_linestyle: "-".into(),
            grid_linewidth: 0.8,
            grid_alpha: 1.0,
            lines_linewidth: 1.5,
            lines_markersize: 6.0,
            lines_markeredgewidth: 1.0,
            lines_dashed_pattern: [3.7, 1.6],
            lines_dashdot_pattern: [6.4, 1.6, 1.0, 1.6],
            lines_dotted_pattern: [1.0, 1.65],
            patch_linewidth: 1.0,
            xtick: Rc::tick_defaults(true),
            ytick: Rc::tick_defaults(false),
            ir: ir::Rc {
                path_simplify: true,
                path_simplify_threshold: 0.111111111111,
                agg_path_chunksize: 0,
                text_hinting: "force_autohint".into(),
                text_hinting_factor: 8,
                text_kerning_factor: 0,
                text_antialiased: true,
                mathtext_fontset: "dejavusans".into(),
                mathtext_fallback: Some("cm".into()),
                mathtext_default: "it".into(),
                mathtext_rm: "sans".into(),
                mathtext_it: "sans:italic".into(),
                mathtext_bf: "sans:bold".into(),
                mathtext_sf: "sans".into(),
                mathtext_tt: "monospace".into(),
                mathtext_cal: "cursive".into(),
                mathtext_bfit: "sans:italic:bold".into(),
                image_resample: true,
                classic_mode: false,
                path_snap: true,
                hatch_color: [0.0, 0.0, 0.0, 1.0],
                hatch_linewidth: 1.0,
            },
        }
    }

    fn set(&mut self, name: &str, value: &RcValue) {
        let number = || match value { RcValue::Number(v) => *v, RcValue::Integer(v) => *v as f64, _ => panic!("rc {name}: number expected") };
        let text = || match value { RcValue::Text(v) => v.to_string(), _ => panic!("rc {name}: text expected") };
        let flag = || match value { RcValue::Flag(v) => *v, _ => panic!("rc {name}: flag expected") };
        let tick = |rc: &mut Rc, prefix: &str| -> Option<()> {
            let (family, field) = name.split_once('.')?;
            if family != prefix { return None; }
            let target = if prefix == "xtick" { &mut rc.xtick } else { &mut rc.ytick };
            match field {
                "major.size" => target.major_size = number(),
                "major.width" => target.major_width = number(),
                "major.pad" => target.major_pad = number(),
                "minor.size" => target.minor_size = number(),
                "minor.width" => target.minor_width = number(),
                "minor.pad" => target.minor_pad = number(),
                "color" => target.color = text(),
                "labelcolor" => target.labelcolor = text(),
                "labelsize" => target.labelsize = number(),
                "direction" => target.direction = text(),
                "top" | "right" => target.tick2 = flag(),
                "bottom" | "left" => target.tick1 = flag(),
                "labeltop" | "labelright" => target.label2 = flag(),
                "labelbottom" | "labelleft" => target.label1 = flag(),
                // Minor tick visibility: the renderer installs no minor locator.
                "minor.top" | "minor.right" | "minor.bottom" | "minor.left" => {}
                _ => panic!("unhandled rc {name}"),
            }
            Some(())
        };
        if tick(self, "xtick").is_some() || tick(self, "ytick").is_some() { return; }
        match name {
            "font.family" => self.font_family = match value { RcValue::Texts(v) => v.iter().map(|s| s.to_string()).collect(), _ => vec![text()] },
            "font.sans-serif" => {}
            "font.size" => self.font_size = number(),
            "font.weight" => self.font_weight = text(),
            "axes.labelsize" => self.axes_labelsize = number(),
            "axes.labelweight" | "axes.titleweight" | "figure.titleweight" | "figure.titlesize" => {}
            "axes.titlesize" => self.axes_titlesize = number(),
            "axes.titlepad" => self.axes_titlepad = number(),
            "axes.labelpad" => self.axes_labelpad = number(),
            "axes.linewidth" => self.axes_linewidth = number(),
            "axes.edgecolor" => self.axes_edgecolor = text(),
            "axes.facecolor" => self.axes_facecolor = text(),
            "axes.titlecolor" => self.axes_titlecolor = text(),
            "axes.labelcolor" => self.axes_labelcolor = text(),
            "axes.grid" => {}
            "figure.facecolor" => self.figure_facecolor = text(),
            "grid.linestyle" => self.grid_linestyle = text(),
            "text.color" => self.text_color = text(),
            "lines.linewidth" => self.lines_linewidth = number(),
            "lines.markersize" => self.lines_markersize = number(),
            "mathtext.fontset" => { self.math_fontfamily = text(); self.ir.mathtext_fontset = text(); }
            "mathtext.fallback" => self.ir.mathtext_fallback = Some(text()),
            "hatch.color" => self.ir.hatch_color = super::color::rgba(&text()),
            // Params no artist of this renderer reads.
            "scatter.edgecolors" | "legend.numpoints" | "legend.fontsize" | "legend.title_fontsize" | "legend.frameon"
            | "figure.subplot.left" | "figure.subplot.right" | "figure.subplot.bottom" | "figure.subplot.top"
            | "figure.autolayout" | "figure.constrained_layout.use" | "text.usetex" | "savefig.facecolor"
            | "savefig.edgecolor" | "savefig.transparent" | "patch.edgecolor" | "patch.force_edgecolor"
            | "lines.linestyle" | "lines.marker" | "image.cmap" | "image.origin" => {}
            _ => panic!("unhandled rc {name}"),
        }
    }

    pub fn tick(&self, x: bool) -> &TickRc { if x { &self.xtick } else { &self.ytick } }
}

impl TickRc {
    fn with_x(mut self, x: bool) -> Self {
        if !x { self.tick1 = true; self.tick2 = false; }
        self
    }
}
