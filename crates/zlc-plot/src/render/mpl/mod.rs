//! The matplotlib object model the renderer keeps: a figure, its axes with
//! their axes, ticks and children, as matplotlib's constructors and setters
//! leave them under the style -- and the resolved IR (`zlc_figure::ir`) they
//! are drawn from.
pub mod artist;
pub mod axes;
pub mod color;
pub mod figure;
pub mod image;
pub mod nodes;
pub mod rc;
pub mod ticker;
