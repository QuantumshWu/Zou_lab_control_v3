//! The shared plot vocabulary (`zlc_plot.kinds`, and the enums of
//! `zlc_plot.specs` every layer names): plot kinds, the three data domains,
//! a reference to one producer-declared axis, reductions and limit modes.
//! Configuration, projection, fit and rendering all use these one types.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlotKind { Curve, Image, Histogram, Rolling, FacetGrid, PulseTimeline }

impl PlotKind {
    pub const ALL: [PlotKind; 6] = [PlotKind::Curve, PlotKind::Image, PlotKind::Histogram, PlotKind::Rolling,
                                    PlotKind::FacetGrid, PlotKind::PulseTimeline];
    pub fn value(self) -> &'static str {
        match self {
            PlotKind::Curve => "curve", PlotKind::Image => "image", PlotKind::Histogram => "histogram",
            PlotKind::Rolling => "rolling", PlotKind::FacetGrid => "facet_grid", PlotKind::PulseTimeline => "pulse_timeline",
        }
    }
    pub fn parse(text: &str) -> Option<Self> { Self::ALL.into_iter().find(|kind| kind.value() == text) }
    /// How many recent shots a new panel of this kind reads (v2 `_window_parameter`):
    /// a Rolling trace its last 100, every other kind its latest event.
    pub fn default_window(self) -> usize { if self == PlotKind::Rolling { 100 } else { 1 } }
}

/// The three axis domains a producer declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisDomain { Repeat, Point, CellData }

impl AxisDomain {
    pub fn value(self) -> &'static str {
        match self { AxisDomain::Repeat => "repeat", AxisDomain::Point => "point", AxisDomain::CellData => "cell_data" }
    }
    pub fn parse(text: &str) -> Option<Self> {
        [AxisDomain::Repeat, AxisDomain::Point, AxisDomain::CellData].into_iter().find(|domain| domain.value() == text)
    }
}

/// A plot-side reference to one producer-declared axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AxisRef { pub domain: AxisDomain, pub axis_id: String }

impl AxisRef {
    /// The axis id is trimmed and must not be empty (`AxisRef.__post_init__`).
    pub fn new(domain: AxisDomain, axis_id: &str) -> Result<Self, String> {
        let axis_id = axis_id.trim();
        if axis_id.is_empty() { return Err(format!("{} requires a non-empty axis_id", domain.value())); }
        Ok(AxisRef { domain, axis_id: axis_id.to_owned() })
    }
    pub fn repeat(axis_id: &str) -> Result<Self, String> { Self::new(AxisDomain::Repeat, axis_id) }
    pub fn point(axis_id: &str) -> Result<Self, String> { Self::new(AxisDomain::Point, axis_id) }
    pub fn cell_data(axis_id: &str) -> Result<Self, String> { Self::new(AxisDomain::CellData, axis_id) }
}

/// How repeated samples that share a plotted coordinate are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reduction { Mean, Sum, Min, Max, First, Last }

impl Reduction {
    pub const ALL: [Reduction; 6] = [Reduction::Mean, Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::First, Reduction::Last];
    pub fn value(self) -> &'static str {
        match self { Reduction::Mean => "mean", Reduction::Sum => "sum", Reduction::Min => "min", Reduction::Max => "max",
                     Reduction::First => "first", Reduction::Last => "last" }
    }
    pub fn parse(text: &str) -> Option<Self> { Self::ALL.into_iter().find(|reduction| reduction.value() == text) }
    /// Last is Scope followed by the ordinary mean, including its sigma.
    pub fn statistic(self) -> Self { if self == Reduction::Last { Reduction::Mean } else { self } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelimMode { Tight, Normal, Fixed }

impl RelimMode {
    pub fn value(self) -> &'static str { match self { RelimMode::Tight => "tight", RelimMode::Normal => "normal", RelimMode::Fixed => "fixed" } }
    pub fn parse(text: &str) -> Option<Self> {
        [RelimMode::Tight, RelimMode::Normal, RelimMode::Fixed].into_iter().find(|mode| mode.value() == text.trim().to_lowercase())
    }
}
