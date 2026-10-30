//! Window chrome that makes the native window open as a zlc_ui FluentWindow:
//! the product icon and zlc_ui's screen-fit size centred on the primary work
//! area (the caption is the page's: frame.rs). Only the frame is touched here;
//! the page draws everything inside it.
use tao::{dpi::PhysicalPosition, platform::windows::{IconExtWindows, WindowExtWindows}, window::{Icon, Window}};

/// zlc_ui/fluent/style.py, the tokens screen_fit_window_size reads.
const AUTO_SCALE_BASIS: (f64, f64) = (1280.0, 790.0);
const AUTO_SCALE_MARGIN: (f64, f64) = (48.0, 88.0);
const FLUENT_SCALE: (f64, f64) = (0.72, 1.25);
const WINDOW_SCREEN_FRACTION: f64 = 0.90;

/// The product mark zlc_ui's windows and taskbar button wear.
pub fn icon() -> Option<Icon> {
    // The executable's own icon resource (build.rs: RT_GROUP_ICON 1).
    Icon::from_resource(1, None).ok()
}

/// Synchronize native minimize state before Tao changes visibility/styles.
/// Otherwise hiding a natively minimized window can save its iconic rectangle
/// as the normal restore position. Keep the same window and page alive.
pub fn set_visible(window: &Window, visible: bool) {
    window.set_minimized(window.is_minimized());
    window.set_visible(visible);
    if visible && window.is_minimized() { window.set_minimized(false); }
}

fn work_area() -> Option<(i32, i32, i32, i32)> {
    use windows_sys::Win32::{Foundation::RECT, UI::WindowsAndMessaging::{SPI_GETWORKAREA, SystemParametersInfoW}};
    let mut rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    // SAFETY: SPI_GETWORKAREA writes one RECT into the pointer it is given.
    let ok = unsafe { SystemParametersInfoW(SPI_GETWORKAREA, 0, (&mut rect as *mut RECT).cast(), 0) };
    (ok != 0).then_some((rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top))
}

/// Size the window as zlc_ui's launcher sizes a FluentWindow and centre it on
/// the primary screen's available area (screen_fit_window_size +
/// center_window_on_primary_screen).  Qt 5 measures in logical px of
/// round-half-up(Windows scale) device px; the result is in device px, so the
/// window covers the same pixels the v2 window does.
pub fn place(window: &Window) { place_fraction(window, WINDOW_SCREEN_FRACTION); }

/// `screen_fit_window_size(fraction)` (a device log opens at 0.45), centred.
pub fn place_fraction(window: &Window, fraction: f64) {
    let Some((target_w, target_h, qt)) = fit_size(window, fraction) else { return };
    set_frame(window, (target_w * qt) as i32, (target_h * qt) as i32);
}

/// v2's snug control window (open_device_control): at least half the standard
/// width, as wide and as tall as its content plus the chrome, at most the
/// screen's width and 0.9 of its height; before the content is measured, half
/// the standard width at the standard height.
pub fn place_snug(window: &Window, content_css: Option<(f64, f64)>) {
    let (Some((target_w, target_h, qt)), Some((_, _, width, height))) = (fit_size(window, WINDOW_SCREEN_FRACTION), work_area()) else { return };
    let half = ((target_w / 2.0).floor() * qt) as i32;
    let (frame_w, frame_h) = match content_css {
        Some((css_w, css_h)) => {
            let (chrome_w, chrome_h) = chrome_size(window).unwrap_or((0, 0));
            let device = |css: f64| (css * window.scale_factor()).ceil() as i32;
            // The content sits under the page's title band, as v2's under its title bar.
            let band = crate::frame::bar_height(window);
            (half.max(device(css_w) + chrome_w).min(width), (device(css_h) + band + chrome_h).min((height as f64 * 0.9) as i32))
        }
        None => (half, (target_h * qt) as i32),
    };
    set_frame(window, frame_w, frame_h);
}

/// The screen-fit target in Qt logical px, and Qt's device ratio.
fn fit_size(window: &Window, fraction: f64) -> Option<(f64, f64, f64)> {
    let (_, _, width, height) = work_area()?;
    let qt = (window.scale_factor() + 0.5).floor().max(1.0);
    let (w, h) = ((width as f64 / qt).floor(), (height as f64 / qt).floor());
    let auto = 1f64.min(((w - AUTO_SCALE_MARGIN.0) / AUTO_SCALE_BASIS.0).max(0.1))
        .min(((h - AUTO_SCALE_MARGIN.1) / AUTO_SCALE_BASIS.1).max(0.1));
    let fluent = auto.clamp(FLUENT_SCALE.0, FLUENT_SCALE.1);
    let scaled = |value: f64, minimum: f64| (value * fluent).round_ties_even().max(minimum);
    let titlebar = scaled(36.0, 28.0);
    let (max_w, max_h) = ((w - scaled(40.0, 28.0)).max(360.0), (h - titlebar - scaled(48.0, 32.0)).max(320.0));
    let (min_w, min_h) = (scaled(980.0, 820.0).min(max_w), scaled(640.0, 560.0).min(max_h));
    let target_w = min_w.max(max_w.min((w * fraction).floor()));
    let target_h = min_h.max(max_h.min((h * fraction).floor() - titlebar));
    Some((target_w, target_h, qt))
}

/// The native caption and edges: the visible frame less the client area.
fn chrome_size(window: &Window) -> Option<(i32, i32)> {
    let inner = window.inner_size();
    let outer = window.outer_position().ok()?;
    let size = window.outer_size();
    let (_, _, visible_w, visible_h) = visible_frame(window).unwrap_or((outer.x, outer.y, size.width as i32, size.height as i32));
    Some(((visible_w - inner.width as i32).max(0), (visible_h - inner.height as i32).max(0)))
}

/// The visible frame at `frame_w`×`frame_h` device px, centred on the work area.
fn set_frame(window: &Window, frame_w: i32, frame_h: i32) {
    let Some((left, top, width, height)) = work_area() else { return };
    // The v2 window's visible frame is this size. Windows' outer rectangle
    // also holds the invisible resize borders, which the visible (DWM
    // extended) frame does not; the caption is the page's (frame.rs), so the
    // outer rectangle is the frame and those borders (no AdjustWindowRect: it
    // would count a caption the window has not got).
    let Ok(outer) = window.outer_position() else { return };
    let size = window.outer_size();
    let (visible_left, visible_top, visible_w, visible_h) = visible_frame(window).unwrap_or((outer.x, outer.y, size.width as i32, size.height as i32));
    let (x, y) = (left + (width - frame_w) / 2 - (visible_left - outer.x), top + (height - frame_h) / 2 - (visible_top - outer.y));
    let (outer_w, outer_h) = (frame_w + size.width as i32 - visible_w, frame_h + size.height as i32 - visible_h);
    use windows_sys::Win32::UI::WindowsAndMessaging::{SetWindowPos, SWP_NOACTIVATE, SWP_NOZORDER};
    // SAFETY: a live window handle; position and size together, the z-order kept.
    unsafe { SetWindowPos(window.hwnd() as windows_sys::Win32::Foundation::HWND, std::ptr::null_mut(), x, y, outer_w.max(1), outer_h.max(1), SWP_NOZORDER | SWP_NOACTIVATE) };
}

/// v2's dialog window (FluentDialogWindow.exec_, the point review): centred on the
/// window it came from, then moved the least it takes to be whole on that screen's
/// work area (_placed_on_screen); a parked run's windows stay parked.
pub fn centre_on(window: &Window, parent: &Window, on_screen: bool) {
    let frame = |window: &Window| visible_frame(window).or_else(|| {
        let at = window.outer_position().ok()?;
        let size = window.outer_size();
        Some((at.x, at.y, size.width as i32, size.height as i32))
    });
    let (Ok(outer), Some((left, top, frame_w, frame_h)), Some((parent_x, parent_y, parent_w, parent_h))) = (window.outer_position(), frame(window), frame(parent)) else { return };
    let (mut x, mut y) = (parent_x + (parent_w - frame_w) / 2, parent_y + (parent_h - frame_h) / 2);
    if let Some((area_x, area_y, area_w, area_h)) = work_area_at(x, y).filter(|_| on_screen) {
        x = x.max(area_x).min((area_x + area_w - frame_w).max(area_x));
        y = y.max(area_y).min((area_y + area_h - frame_h).max(area_y));
    }
    window.set_outer_position(PhysicalPosition::new(x - (left - outer.x), y - (top - outer.y)));
}

/// The work area of the screen a point is on (the nearest one if none).
fn work_area_at(x: i32, y: i32) -> Option<(i32, i32, i32, i32)> {
    use windows_sys::Win32::{Foundation::POINT, Graphics::Gdi::{GetMonitorInfoW, MonitorFromPoint, MONITORINFO, MONITOR_DEFAULTTONEAREST}};
    // SAFETY: MonitorFromPoint takes a point by value; GetMonitorInfoW fills a MONITORINFO whose size is set.
    unsafe {
        let monitor = MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTONEAREST);
        let mut info: MONITORINFO = std::mem::zeroed();
        info.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
        if GetMonitorInfoW(monitor, &mut info) == 0 { return None; }
        let area = info.rcWork;
        Some((area.left, area.top, area.right - area.left, area.bottom - area.top))
    }
}

fn visible_frame(window: &Window) -> Option<(i32, i32, i32, i32)> {
    use windows_sys::Win32::{Foundation::RECT, Graphics::Dwm::{DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute}};
    let mut rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    let hwnd = window.hwnd() as windows_sys::Win32::Foundation::HWND;
    // SAFETY: a live window handle and a RECT-sized output, as the attribute documents.
    let status = unsafe { DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS as u32, (&mut rect as *mut RECT).cast(), std::mem::size_of::<RECT>() as u32) };
    (status == 0 && rect.right > rect.left).then_some((rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top))
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires a Windows desktop session; uses an isolated offscreen window"]
    fn native_minimize_hide_reopen_preserves_normal_rectangle() {
        use super::*;
        use std::time::{Duration, Instant};
        use tao::{event::Event, event_loop::{ControlFlow, EventLoopBuilder},
            platform::{run_return::EventLoopExtRunReturn, windows::{EventLoopBuilderExtWindows, WindowBuilderExtWindows}},
            window::WindowBuilder, dpi::LogicalSize};
        use windows_sys::Win32::UI::WindowsAndMessaging::{ShowWindow, SW_SHOWMINNOACTIVE};
        let mut events = EventLoopBuilder::new().with_any_thread(true).build();
        let window = WindowBuilder::new().with_visible(false).with_focused(false).with_skip_taskbar(true)
            .with_inner_size(LogicalSize::new(900.0, 650.0)).build(&events).unwrap();
        crate::frame::install(&window, |_| {}).unwrap();
        window.set_outer_position(PhysicalPosition::new(-20000, -20000));
        let original = window.inner_size();
        let mut step = 0;
        let mut next = Instant::now();
        let mut wrong = Vec::new();
        events.run_return(|event, _, flow| {
            *flow = ControlFlow::WaitUntil(next);
            if !matches!(event, Event::MainEventsCleared) || Instant::now() < next { return; }
            match step % 5 {
                0 => set_visible(&window, true),
                1 => { unsafe { ShowWindow(window.hwnd() as _, SW_SHOWMINNOACTIVE); } },
                2 => set_visible(&window, false),
                3 => set_visible(&window, true),
                _ => { if window.is_minimized() || !window.is_visible() || window.inner_size() != original { wrong.push(window.inner_size()); } },
            }
            step += 1;
            next = Instant::now() + Duration::from_millis(20);
            if step == 50 { set_visible(&window, false); *flow = ControlFlow::Exit; }
        });
        assert!(wrong.is_empty(), "normal rectangle {original:?} changed to {wrong:?}");
    }
}

