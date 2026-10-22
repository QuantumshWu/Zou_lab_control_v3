//! v2's frameless FluentWindow (qframelesswindow 0.8 under zlc_ui) on a native
//! window. The client is the whole window, as WindowsFramelessWindow's
//! WM_NCCALCSIZE makes it (a maximized one inset by its frame), the DWM shadow
//! extends over it and Windows 11 rounds it; the page draws v2's 32 Qt px title
//! bar (the mark, the title, the minimize, maximize and close buttons). The
//! window keeps its caption styles, so snap, the minimize/maximize animations
//! and the system menu stay the system's.
//!
//! The page covers the whole client, so hit windows above it (no surface of
//! their own: the page shows through) answer v2's hit tests where v2 answered
//! them: its 5 device px resize border on every side, and in the title band
//! v2's three 46 Qt px buttons (HTMAXBUTTON, so Windows 11 offers its snap
//! layouts there) and the caption between. Moves, sizes, double clicks and the
//! system menu are the frame's own; a button's hover and press are told to the
//! page (`{type: "caption", hover, pressed, maximized}`), which paints them as
//! v2's TitleBarButton does.
use std::sync::OnceLock;

use serde_json::{json, Value};
use tao::{platform::windows::WindowExtWindows, window::Window};
use windows_sys::Win32::{
    Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM},
    Graphics::{Dwm::{DwmExtendFrameIntoClientArea, DwmSetWindowAttribute, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND}, Gdi::ScreenToClient},
    System::LibraryLoader::GetModuleHandleW,
    UI::{
        Controls::MARGINS,
        HiDpi::{GetDpiForWindow, GetSystemMetricsForDpi},
        Input::KeyboardAndMouse::{TrackMouseEvent, TME_LEAVE, TME_NONCLIENT, TRACKMOUSEEVENT},
        Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass},
        WindowsAndMessaging::*,
    },
};

/// TitleBarBase.setFixedHeight(32) and TitleBarButton.setFixedSize(46, 32), in Qt px.
const BAR_QT: i32 = 32;
const BUTTON_QT: i32 = 46;
/// WindowsFramelessWindow.BORDER_WIDTH: device px, compared as v2 compares them.
const BORDER: i32 = 5;
const SUBCLASS: usize = 0x5a4c43;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Button { Minimize, Maximize, Close }

impl Button {
    fn of(hit: usize) -> Option<Button> {
        match hit as u32 { HTMINBUTTON => Some(Button::Minimize), HTMAXBUTTON => Some(Button::Maximize), HTCLOSE => Some(Button::Close), _ => None }
    }
    fn word(self) -> &'static str { match self { Button::Minimize => "minimize", Button::Maximize => "maximize", Button::Close => "close" } }
}

/// The window's hit windows: the title band, then the left, right and bottom borders.
struct Frame {
    parent: HWND,
    hits: [HWND; 4],
    notify: Box<dyn Fn(Value)>,
    hover: Option<Button>,
    pressed: Option<Button>,
    maximized: bool,
    tracking: bool,
}

impl Frame {
    fn message(&self) -> Value {
        // The client's width: the page's own can be a device pixel or two wider (Chromium ceils it
        // to whole CSS px), and v2's buttons stand at the window's right edge.
        json!({"type": "caption", "hover": self.hover.map(Button::word), "pressed": self.pressed.map(Button::word), "maximized": self.maximized,
               "width": client(self.parent).0})
    }
    fn tell(&self) { (self.notify)(self.message()); }
    fn set(&mut self, hover: Option<Button>, pressed: Option<Button>) {
        if (hover, pressed) != (self.hover, self.pressed) { (self.hover, self.pressed) = (hover, pressed); self.tell(); }
    }
}

/// Qt 5's device ratio for the window's monitor (round half up), the unit of v2's Qt px.
fn qt_ratio(hwnd: HWND) -> i32 {
    // SAFETY: a live window handle.
    let dpi = unsafe { GetDpiForWindow(hwnd) }.max(96);
    ((dpi as f64 / 96.0) + 0.5).floor() as i32
}

/// The title band in device px of the window's monitor: what the page draws there.
pub fn bar_height(window: &Window) -> i32 { BAR_QT * qt_ratio(window.hwnd() as HWND) }

fn zoomed(hwnd: HWND) -> bool {
    // SAFETY: a live window handle.
    (unsafe { IsZoomed(hwnd) }) != 0
}

fn client(hwnd: HWND) -> (i32, i32) {
    let mut rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    // SAFETY: a live window handle and a RECT to fill.
    unsafe { GetClientRect(hwnd, &mut rect) };
    (rect.right - rect.left, rect.bottom - rect.top)
}

/// The hit windows over the band and the borders, above the page; none on a maximized window's borders.
fn place(frame: &Frame) {
    let (width, height) = client(frame.parent);
    let bar = BAR_QT * qt_ratio(frame.parent);
    let edges = !zoomed(frame.parent);
    let rects = [(0, 0, width, bar), (0, bar, BORDER, height - bar), (width - BORDER + 1, bar, BORDER - 1, height - bar), (0, height - BORDER + 1, width, BORDER - 1)];
    for (index, (&hwnd, (x, y, w, h))) in frame.hits.iter().zip(rects).enumerate() {
        let shown = if index == 0 || edges { SWP_SHOWWINDOW } else { SWP_HIDEWINDOW };
        // SAFETY: children of the live parent.
        unsafe { SetWindowPos(hwnd, HWND_TOP, x, y, w.max(0), h.max(0), SWP_NOACTIVATE | shown) };
    }
}

/// Make `window` v2's frameless FluentWindow; `notify` hears the caption buttons' state for the page.
pub fn install(window: &Window, notify: impl Fn(Value) + 'static) -> Result<(), String> {
    let parent = window.hwnd() as HWND;
    let class = hit_class();
    let mut hits = [std::ptr::null_mut(); 4];
    for hit in &mut hits {
        // SAFETY: a child of a live window, of a registered class; no surface of its own (the page shows through).
        *hit = unsafe {
            CreateWindowExW(WS_EX_LAYERED | WS_EX_NOREDIRECTIONBITMAP, class, std::ptr::null(), WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS,
                            0, 0, 0, 0, parent, std::ptr::null_mut(), GetModuleHandleW(std::ptr::null()), std::ptr::null())
        };
        if hit.is_null() { return Err(format!("Cannot make the window's frame: {}", std::io::Error::last_os_error())); }
    }
    let frame = Box::into_raw(Box::new(Frame { parent, hits, notify: Box::new(notify), hover: None, pressed: None, maximized: false, tracking: false }));
    // SAFETY: the frame lives until the parent's WM_NCDESTROY frees it (parent_proc); the hit windows die with the parent.
    unsafe {
        for &hit in &(*frame).hits {
            SetLayeredWindowAttributes(hit, 0, 255, LWA_ALPHA);
            SetWindowLongPtrW(hit, GWLP_USERDATA, frame as isize);
        }
        if SetWindowSubclass(parent, Some(parent_proc), SUBCLASS, frame as usize) == 0 {
            return Err("Cannot take over the window's frame".into());
        }
        // addShadowEffect: the DWM frame over the whole client; Windows 11 rounds the window.
        let margins = MARGINS { cxLeftWidth: -1, cxRightWidth: -1, cyTopHeight: -1, cyBottomHeight: -1 };
        DwmExtendFrameIntoClientArea(parent, &margins);
        let round = DWMWCP_ROUND;
        DwmSetWindowAttribute(parent, DWMWA_WINDOW_CORNER_PREFERENCE as u32, (&round as *const i32).cast(), 4);
        SetWindowPos(parent, std::ptr::null_mut(), 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        place(&*frame);
    }
    Ok(())
}

fn frame_of(window: &Window) -> Option<&'static Frame> {
    // SAFETY: any of the parent's hit windows names the frame, which lives as long as the parent.
    unsafe {
        let hit = FindWindowExW(window.hwnd() as HWND, std::ptr::null_mut(), hit_class(), std::ptr::null());
        if hit.is_null() { return None; }
        (GetWindowLongPtrW(hit, GWLP_USERDATA) as *const Frame).as_ref()
    }
}

/// Keep the hit windows above the page once the page's own window exists.
pub fn raise(window: &Window) { if let Some(frame) = frame_of(window) { place(frame); } }

/// The title bar's state for a page that has just loaded.
pub fn caption(window: &Window) -> Option<Value> { frame_of(window).map(Frame::message) }

fn hit_class() -> *const u16 {
    static CLASS: OnceLock<Vec<u16>> = OnceLock::new();
    CLASS.get_or_init(|| {
        let name: Vec<u16> = "ZlcFrameHit".encode_utf16().chain([0]).collect();
        // SAFETY: a class of this module whose procedure is hit_proc.
        unsafe {
            let class = WNDCLASSEXW { cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32, style: 0, lpfnWndProc: Some(hit_proc),
                cbClsExtra: 0, cbWndExtra: 0, hInstance: GetModuleHandleW(std::ptr::null()), hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW), hbrBackground: std::ptr::null_mut(), lpszMenuName: std::ptr::null(),
                lpszClassName: name.as_ptr(), hIconSm: std::ptr::null_mut() };
            RegisterClassExW(&class);
        }
        name
    }).as_ptr()
}

unsafe extern "system" fn parent_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM, _id: usize, data: usize) -> LRESULT {
    let frame = data as *mut Frame;
    match message {
        WM_NCCALCSIZE if wparam != 0 => {
            // The whole window is client; a maximized one hangs its frame over the monitor's edges.
            if zoomed(hwnd) {
                let rect = unsafe { &mut (*(lparam as *mut NCCALCSIZE_PARAMS)).rgrc[0] };
                // SAFETY: metrics of the window's own DPI.
                let (x, y) = unsafe {
                    let dpi = GetDpiForWindow(hwnd);
                    let padded = GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
                    (GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) + padded, GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi) + padded)
                };
                rect.left += x; rect.right -= x; rect.top += y; rect.bottom -= y;
            }
            return 0;
        }
        WM_SIZE => {
            let frame = unsafe { &mut *frame };
            place(frame);
            if wparam as u32 != SIZE_MINIMIZED {
                let maximized = wparam as u32 == SIZE_MAXIMIZED;
                // MaximizeButton.setMaxState: a new glyph starts in its normal face.
                if maximized != frame.maximized { (frame.maximized, frame.hover, frame.pressed) = (maximized, None, None); }
                frame.tell();
            }
        }
        WM_NCDESTROY => {
            unsafe { RemoveWindowSubclass(hwnd, Some(parent_proc), SUBCLASS) };
            let result = unsafe { DefSubclassProc(hwnd, message, wparam, lparam) };
            // SAFETY: made by install, freed once, after the hit windows (children) are gone.
            drop(unsafe { Box::from_raw(frame) });
            return result;
        }
        _ => {}
    }
    unsafe { DefSubclassProc(hwnd, message, wparam, lparam) }
}

/// WindowsFramelessWindow.nativeEvent's borders first, then the title bar's buttons and caption.
fn hit(frame: &Frame, lparam: LPARAM) -> u32 {
    let mut point = POINT { x: (lparam & 0xffff) as i16 as i32, y: ((lparam >> 16) & 0xffff) as i16 as i32 };
    // SAFETY: a live window handle and a point to convert.
    unsafe { ScreenToClient(frame.parent, &mut point) };
    let (width, height) = client(frame.parent);
    let border = if zoomed(frame.parent) { 0 } else { BORDER };
    let (left, right, top, bottom) = (point.x < border, point.x > width - border, point.y < border, point.y > height - border);
    match (left, right, top, bottom) {
        (true, _, true, _) => return HTTOPLEFT,
        (_, true, _, true) => return HTBOTTOMRIGHT,
        (_, true, true, _) => return HTTOPRIGHT,
        (true, _, _, true) => return HTBOTTOMLEFT,
        (_, _, true, _) => return HTTOP,
        (_, _, _, true) => return HTBOTTOM,
        (true, _, _, _) => return HTLEFT,
        (_, true, _, _) => return HTRIGHT,
        _ => {}
    }
    if point.y >= BAR_QT * qt_ratio(frame.parent) { return HTCLIENT; }
    let button = BUTTON_QT * qt_ratio(frame.parent);
    if point.x >= width - button { HTCLOSE } else if point.x >= width - 2 * button { HTMAXBUTTON } else if point.x >= width - 3 * button { HTMINBUTTON } else { HTCAPTION }
}

unsafe extern "system" fn hit_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let frame = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut Frame;
    if frame.is_null() { return unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }; }
    // SAFETY: set by install; the frame outlives its hit windows.
    let frame = unsafe { &mut *frame };
    match message {
        WM_NCHITTEST => return hit(frame, lparam) as LRESULT,
        WM_NCMOUSEMOVE => {
            if !frame.tracking {
                let mut track = TRACKMOUSEEVENT { cbSize: std::mem::size_of::<TRACKMOUSEEVENT>() as u32, dwFlags: TME_LEAVE | TME_NONCLIENT, hwndTrack: hwnd, dwHoverTime: 0 };
                frame.tracking = unsafe { TrackMouseEvent(&mut track) } != 0;
            }
            frame.set(Button::of(wparam), frame.pressed);
            return 0;
        }
        WM_NCMOUSELEAVE => { frame.tracking = false; frame.set(None, None); return 0; }
        WM_NCLBUTTONDOWN | WM_NCLBUTTONDBLCLK => match Button::of(wparam) {
            Some(button) => { frame.set(Some(button), Some(button)); return 0; }
            // The caption and the borders are the frame's: move, size, double-click maximize.
            None => return unsafe { SendMessageW(frame.parent, message, wparam, lparam) },
        },
        WM_NCLBUTTONUP => {
            let (button, pressed) = (Button::of(wparam), frame.pressed);
            frame.set(button, None);
            if let Some(button) = button.filter(|button| Some(*button) == pressed) {
                // SAFETY: the parent is live; each is what the system's own caption button does.
                unsafe {
                    match button {
                        Button::Minimize => { ShowWindow(frame.parent, SW_MINIMIZE); }
                        Button::Maximize => { ShowWindow(frame.parent, if zoomed(frame.parent) { SW_RESTORE } else { SW_MAXIMIZE }); }
                        Button::Close => { PostMessageW(frame.parent, WM_SYSCOMMAND, SC_CLOSE as usize, 0); }
                    }
                }
            }
            return 0;
        }
        // The system menu on the caption, as on any window's.
        WM_NCRBUTTONDOWN | WM_NCRBUTTONUP | WM_NCRBUTTONDBLCLK if Button::of(wparam).is_none() => return unsafe { SendMessageW(frame.parent, message, wparam, lparam) },
        _ => {}
    }
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}
