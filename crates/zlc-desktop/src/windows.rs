//! v2's top-level windows -- Devices@Zou lab, TaskConsole@Zou lab,
//! PulseGUI@Zou lab, one FigureViewer@Zou lab per opened Figure, and each
//! device's own control and log window -- each a native window with its own
//! WebView2 page (`index.html?window=<key>&base=<first request>`), its own
//! document's state and its own plot fronts, all sharing the one core. The
//! window opening first is Devices (v2's experiment entry); every other is made
//! when it is first shown, as v2 made its windows, on the first one's WebView2
//! environment.
use std::path::PathBuf;

use serde_json::Value;
use tao::{dpi::{LogicalSize, PhysicalPosition}, event_loop::{EventLoopProxy, EventLoopWindowTarget},
          platform::windows::{WindowBuilderExtWindows, WindowExtWindows}, window::{Window, WindowBuilder, WindowId}};
use webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2Environment;
use wry::{WebContext, WebView, WebViewBuilder, WebViewBuilderExtWindows, WebViewExtWindows};

use crate::{chrome, frame, pixels::Pixels, science::Front, AppEvent};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Kind {
    Devices, TaskConsole, PulseEditor,
    /// One per opened Figure, numbered as opened (its document is the same number).
    FigureViewer(u32),
    /// A device's control window (v2 open_device_control) and its log window.
    Control(String), Log(String),
    /// A run's point review (v2 PointReviewView in its FluentDialogWindow), by node.
    Review(String),
    /// An SLM's Control: its SLM Editor (v2 open_slm_control), by device.
    SlmEditor(String),
    /// installation.exe's one window (installer.rs).
    Installer,
}

const VIEWER: &str = "figure-viewer-";

impl Kind {
    /// The name MCP and acceptance scripts address the window by.
    pub fn key(&self) -> String {
        match self {
            Kind::Devices => "devices".into(), Kind::TaskConsole => "task-console".into(), Kind::PulseEditor => "pulse-editor".into(),
            Kind::FigureViewer(n) => format!("{VIEWER}{n}"), Kind::Control(device) => format!("control:{device}"), Kind::Log(device) => format!("log:{device}"),
            Kind::Review(node) => format!("review:{node}"), Kind::SlmEditor(device) => format!("slm-editor:{device}"),
            Kind::Installer => "installation".into(),
        }
    }
    pub fn parse(key: &str) -> Option<Kind> {
        match key {
            "devices" => Some(Kind::Devices), "task-console" => Some(Kind::TaskConsole), "pulse-editor" => Some(Kind::PulseEditor),
            "installation" => Some(Kind::Installer),
            _ => key.strip_prefix(VIEWER).and_then(|n| n.parse().ok()).map(Kind::FigureViewer)
                .or_else(|| key.strip_prefix("control:").filter(|device| !device.is_empty()).map(|device| Kind::Control(device.into())))
                .or_else(|| key.strip_prefix("log:").filter(|device| !device.is_empty()).map(|device| Kind::Log(device.into())))
                .or_else(|| key.strip_prefix("review:").filter(|node| !node.is_empty()).map(|node| Kind::Review(node.into())))
                .or_else(|| key.strip_prefix("slm-editor:").filter(|device| !device.is_empty()).map(|device| Kind::SlmEditor(device.into()))),
        }
    }
    /// v2's window titles; a device's windows are named by its page (`<role> control`, `<role> log@Zou lab`).
    pub fn title(&self) -> String {
        match self {
            Kind::Devices => "Devices@Zou lab".into(), Kind::TaskConsole => "TaskConsole@Zou lab".into(), Kind::PulseEditor => "PulseGUI@Zou lab".into(),
            Kind::FigureViewer(_) => "FigureViewer@Zou lab".into(), Kind::Control(device) => format!("{device} control"), Kind::Log(device) => format!("{device} log@Zou lab"),
            // Until its page names it with the request's own title.
            Kind::Review(_) => "Review".into(),
            Kind::SlmEditor(device) => format!("{device} SLM Editor"),
            Kind::Installer => "Installation@Zou lab".into(),
        }
    }
    /// The page's view and what it is of.
    fn page(&self) -> String {
        match self {
            Kind::FigureViewer(n) => format!("figure-viewer&document={n}"),
            Kind::Control(device) => format!("control&device={}", encode(device)),
            Kind::Log(device) => format!("log&device={}", encode(device)),
            Kind::Review(node) => format!("review&node={}", encode(node)),
            Kind::SlmEditor(device) => format!("slm-editor&device={}", encode(device)),
            other => other.key(),
        }
    }
    /// The state sections its view reads: its own document, the workbench's for the bench windows.
    /// A Figure Viewer is sent its document by itself (main.rs viewer_message).
    fn reads(&self, section: &str) -> bool {
        match self {
            Kind::TaskConsole => !matches!(section, "pulse" | "viewer" | "viewers"),
            Kind::Devices => !matches!(section, "pulse" | "viewer" | "viewers" | "panels" | "signals" | "document" | "acquisition" | "plot_kinds"),
            Kind::PulseEditor => section == "pulse",
            Kind::Control(_) | Kind::Log(_) => section == "devices",
            Kind::FigureViewer(_) => false,
            Kind::Review(_) => matches!(section, "logic" | "panels" | "plot_kinds" | "signals"),
            // Its own pictures and state come by themselves (main.rs slm_host); the device row says its role.
            Kind::SlmEditor(_) => section == "devices",
            Kind::Installer => section.starts_with("installation"),
        }
    }
}

fn encode(text: &str) -> String { text.bytes().map(|byte| if byte.is_ascii_alphanumeric() || b"-_.".contains(&byte) { (byte as char).to_string() } else { format!("%{byte:02X}") }).collect() }

/// Which window draws a panel's fronts: a Figure's `figure-<n>-…` panels its viewer's.
pub fn owner(panel: &str) -> Kind {
    if panel == crate::pulse_host::PREVIEW_PANEL { return Kind::PulseEditor; }
    if let Some(node) = panel.strip_prefix(crate::panels::REVIEW_PREFIX) { return Kind::Review(node.into()); }
    if let Some(device) = crate::slm_host::device_of(panel) { return Kind::SlmEditor(device.into()); }
    match crate::panels::viewer_of(panel) { Some(n) => Kind::FigureViewer(n), None => Kind::TaskConsole }
}

/// Fields drop in order: the page's mappings, the page, then its window.
pub struct AppWindow { pub kind: Kind, pub pixels: Pixels, _webview: WebView, pub window: Window, pub ready: bool, first_request: u64 }

impl AppWindow {
    fn owns_request(&self, request: u64) -> bool { (self.first_request..self.first_request + REQUESTS).contains(&request) }
}

/// Each window numbers its commands from its own base, so answers never cross windows
/// (remote clients' start at remote::FIRST_REMOTE_REQUEST, past every window's).
const REQUESTS: u64 = 1_000_000_000_000;

/// What every window is made with.
pub struct Setup { pub ui_root: PathBuf, pub browser_args: String, pub devtools: bool, pub parked: Option<PhysicalPosition<i32>> }

/// Browser sharing (share.rs) shows the Task Console window: its states and its pictures.
/// Every page shares the first one's WebView2 environment (its browser process): a window made later is only its controller.
pub struct Windows { open: Vec<AppWindow>, context: WebContext, setup: Setup, proxy: EventLoopProxy<AppEvent>, share: Option<crate::share::Share>, made: u64,
                     environment: Option<ICoreWebView2Environment> }

impl Windows {
    pub fn new(context: WebContext, setup: Setup, proxy: EventLoopProxy<AppEvent>) -> Self { Self { open: Vec::new(), context, setup, proxy, share: None, made: 0, environment: None } }

    pub fn set_share(&mut self, share: crate::share::Share) { self.share = Some(share); }

    /// Make `kind`'s window (shown or hidden) unless it exists.
    pub fn make(&mut self, kind: &Kind, target: &EventLoopWindowTarget<AppEvent>, visible: bool) -> Result<(), String> {
        if self.get(kind).is_some() { return Ok(()); }
        if self.made + 1 >= crate::remote::FIRST_REMOTE_REQUEST / REQUESTS { return Err("Too many windows were opened in this session".into()); }
        let _made = crate::trace::span("window.make", || serde_json::json!({"window": kind.key(), "visible": visible}));
        self.made += 1;
        let first_request = self.made * REQUESTS;
        let window = WindowBuilder::new().with_title(kind.title()).with_visible(false)
            .with_window_icon(chrome::icon()).with_taskbar_icon(chrome::icon())
            .with_inner_size(LogicalSize::new(1160.0, 820.0)).build(target).map_err(|error| error.to_string())?;
        // v2's frameless FluentWindow: the page draws the title bar, the frame keeps the rest.
        let (caption_proxy, caption_kind) = (self.proxy.clone(), kind.clone());
        frame::install(&window, move |message| { let _ = caption_proxy.send_event(AppEvent::Caption { window: caption_kind.clone(), message }); })?;
        // v2's sizes: a screen fit, a device log at 0.45 of it, a control half its width (then its content's height),
        // each centred on the screen; a point review is a dialog, centred on the Task Console it came from.
        match kind {
            Kind::Log(_) => chrome::place_fraction(&window, 0.45),
            Kind::Control(_) => chrome::place_snug(&window, None),
            Kind::Installer => chrome::place_fraction(&window, 0.75),
            _ => chrome::place(&window),
        }
        let parent = matches!(kind, Kind::Review(_)).then(|| self.get(&Kind::TaskConsole)).flatten().filter(|parent| parent.window.is_visible());
        if let Some(parent) = parent { chrome::centre_on(&window, &parent.window, self.setup.parked.is_none()); }
        // Automated runs park every window off the screens (ZLC_V3_WINDOW_AT=x,y); a dialog on its parked parent already is.
        if let Some(position) = self.setup.parked.filter(|_| parent.is_none()) { window.set_outer_position(position); }
        window.set_visible(visible);
        let root = self.setup.ui_root.clone();
        let (title_proxy, command_proxy) = (self.proxy.clone(), self.proxy.clone());
        let (title_kind, command_kind) = (kind.clone(), kind.clone());
        let builder = WebViewBuilder::new_with_web_context(&mut self.context)
            .with_devtools(self.setup.devtools)
            .with_custom_protocol("zlc".into(), move |_webview, request| serve(&root, &request))
            .with_navigation_handler(|url| url.starts_with("http://zlc.localhost/") || url.starts_with("zlc://localhost/"))
            // The page names what it shows after v2's title; the caption says it.
            .with_document_title_changed_handler(move |title| { let _ = title_proxy.send_event(AppEvent::Title { window: title_kind.clone(), title }); })
            .with_ipc_handler(move |request| {
                let event = match serde_json::from_str(request.body()) {
                    Ok(command) => AppEvent::Command { window: Some(command_kind.clone()), command },
                    Err(error) => AppEvent::InvalidMessage(error.to_string()),
                };
                let _ = command_proxy.send_event(event);
            })
            // Every page of one environment is made with the same arguments.
            .with_additional_browser_args(&self.setup.browser_args).with_browser_accelerator_keys(false)
            .with_url(format!("zlc://localhost/index.html?window={}&base={first_request}", kind.page()));
        let builder = match &self.environment { Some(environment) => builder.with_environment(environment.clone()), None => builder };
        let webview = { let _built = crate::trace::span("window.webview", || serde_json::json!({"window": kind.key()})); builder.build(&window).map_err(|error| error.to_string())? };
        if self.environment.is_none() { self.environment = Some(webview.environment()); }
        // The caption window above the page's own.
        frame::raise(&window);
        let mut pixels = Pixels::new(&webview)?;
        if let Some(share) = self.share.as_ref().filter(|_| *kind == Kind::TaskConsole) { pixels.set_share(share.clone()); }
        self.open.push(AppWindow { kind: kind.clone(), pixels, _webview: webview, window, ready: false, first_request });
        Ok(())
    }

    /// Show `kind` in front (made now if it was not). It is shown at once and
    /// made the active window on the next turn (`activate`): activating it --
    /// the focus moving into its page -- took 20-40 ms of this turn, before
    /// the answer to whoever asked could go out.
    pub fn show(&mut self, kind: &Kind, target: &EventLoopWindowTarget<AppEvent>) -> Result<(), String> {
        crate::trace::point("window.show", || serde_json::json!({"window": kind.key(), "made": self.get(kind).is_some()}));
        self.make(kind, target, true)?;
        let window = &self.get(kind).expect("made").window;
        if window.is_minimized() { window.set_minimized(false); }
        use windows_sys::Win32::UI::WindowsAndMessaging::{SW_SHOWNA, ShowWindow};
        unsafe { ShowWindow(window.hwnd() as _, SW_SHOWNA); }
        let _ = self.proxy.send_event(AppEvent::Activate(kind.clone()));
        Ok(())
    }

    /// The second half of `show`, unless the window was hidden again meanwhile.
    pub fn activate(&self, kind: &Kind) {
        if let Some(window) = self.get(kind).map(|window| &window.window).filter(|window| window.is_visible()) {
            window.set_visible(true);
            window.set_focus();
        }
    }

    /// Whether any window is on screen (the application ends with its last, as v2's did).
    pub fn any_visible(&self) -> bool { self.open.iter().any(|window| window.window.is_visible()) }

    pub fn hide(&self, kind: &Kind) { if let Some(window) = self.get(kind) { window.window.set_visible(false); } }
    /// v2's review dialog was application modal: while `modal` is open the other windows take no input.
    pub fn set_modal(&self, modal: &Kind, on: bool) {
        for window in self.open.iter().filter(|window| window.kind != *modal) { window.window.set_enable(!on); }
    }
    /// The window and its page are gone (v2: a closed log, a closed Figure).
    pub fn close(&mut self, kind: &Kind) { self.open.retain(|window| window.kind != *kind); }
    pub fn get(&self, kind: &Kind) -> Option<&AppWindow> { self.open.iter().find(|window| window.kind == *kind) }
    pub fn get_mut(&mut self, kind: &Kind) -> Option<&mut AppWindow> { self.open.iter_mut().find(|window| window.kind == *kind) }
    pub fn kinds(&self) -> impl Iterator<Item = &Kind> { self.open.iter().map(|window| &window.kind) }
    pub fn kind_of(&self, id: WindowId) -> Option<Kind> { self.open.iter().find(|window| window.window.id() == id).map(|window| window.kind.clone()) }
    /// A dialog's parent: the window that asked, else the one in front.
    pub fn hwnd(&self, kind: Option<&Kind>) -> isize {
        kind.and_then(|kind| self.get(kind)).or_else(|| self.open.iter().find(|window| window.window.is_visible())).or(self.open.first())
            .map_or(0, |window| window.window.hwnd())
    }
    pub fn describe(&self) -> Value {
        Value::Array(self.open.iter().map(|window| { let size = window.window.inner_size();
            serde_json::json!({"window": window.kind.key(), "title": window.window.title(), "visible": window.window.is_visible(),
                "size": [size.width, size.height]}) }).collect())
    }

    /// A control window as tall as its content (v2's snug fit), once the page has measured it.
    pub fn fit(&self, kind: &Kind, content_width: f64, content_height: f64) {
        if let Some(window) = self.get(kind) {
            chrome::place_snug(&window.window, Some((content_width, content_height)));
            if let Some(position) = self.setup.parked { window.window.set_outer_position(position); }
        }
    }

    /// One state message to every page, each told what changed in the
    /// sections its view reads (delta.rs); a page with nothing new in it and
    /// no request of its own is not sent it.
    pub fn send(&self, state: &Value) -> Result<(), String> {
        if let Some(share) = self.share.as_ref().filter(|_| state["type"] == "state") { share.publish_state(state); }
        for window in self.open.iter().filter(|window| window.ready) {
            window.pixels.send_state(state, |section| window.kind.reads(section), |request| request.as_u64().is_some_and(|request| window.owns_request(request)))?;
        }
        Ok(())
    }

    /// A state for one page, as `send` would give it (a page just up).
    pub fn send_state_to(&self, kind: &Kind, state: &Value) -> Result<(), String> {
        match self.get(kind).filter(|window| window.ready) {
            Some(window) => window.pixels.send_state(state, |section| window.kind.reads(section), |request| request.as_u64().is_some_and(|request| window.owns_request(request))),
            None => Ok(()),
        }
    }

    /// A message for one page whole (its document, a reply only it asked for): always sent.
    pub fn send_to(&self, kind: &Kind, value: &Value) -> Result<(), String> {
        if let Some(share) = self.share.as_ref().filter(|_| *kind == Kind::TaskConsole && value["type"] == "state") { share.publish_state(value); }
        match self.get(kind).filter(|window| window.ready) { Some(window) => window.pixels.send_state(value, |_| true, |_| true), None => Ok(()) }
    }

    /// `kind`'s page was (re)loaded: it holds no state and no pixel mapping yet.
    pub fn page_loaded(&mut self, kind: &Kind) { if let Some(window) = self.get_mut(kind) { window.pixels.page_loaded(); } }

    /// A panel's front goes to the window that shows the panel.
    pub fn submit(&mut self, panel: String, sequence: u64, front: Front) -> Result<(), String> {
        match self.get_mut(&owner(&panel)).filter(|window| window.ready) { Some(window) => window.pixels.submit(panel, sequence, front), None => Ok(()) }
    }

    /// Each window keeps the mappings of its own panels in `kept`.
    pub fn retain_panels(&mut self, kept: &[String]) -> Result<(), String> {
        for window in self.open.iter_mut().filter(|window| window.ready) {
            let own: Vec<String> = kept.iter().filter(|panel| owner(panel) == window.kind).cloned().collect();
            window.pixels.retain_panels(&own)?;
        }
        Ok(())
    }
}

fn serve(root: &PathBuf, request: &wry::http::Request<Vec<u8>>) -> wry::http::Response<std::borrow::Cow<'static, [u8]>> {
    let relative = request.uri().path().trim_start_matches('/');
    let relative = if relative.is_empty() { "index.html" } else { relative };
    // The product mark the title bar wears, the window's and the taskbar's own file.
    if relative == "zlc.ico" && request.method() == "GET" {
        return wry::http::Response::builder().header("Content-Type", "image/x-icon").body(std::borrow::Cow::Borrowed(&include_bytes!("../assets/zlc.ico")[..])).unwrap();
    }
    match root.join(relative).canonicalize() {
        Ok(path) if path.starts_with(root) && path.is_file() && request.method() == "GET" => {
            let mime = match path.extension().and_then(|x| x.to_str()).unwrap_or("") {
                "html" => "text/html; charset=utf-8", "js" => "text/javascript; charset=utf-8",
                "css" => "text/css; charset=utf-8", "ttf" => "font/ttf", "woff2" => "font/woff2", "ico" => "image/x-icon",
                _ => "application/octet-stream",
            };
            match std::fs::read(path) {
                Ok(data) => wry::http::Response::builder().header("Content-Type", mime).header("Cache-Control", "no-cache").body(data.into()).unwrap(),
                Err(error) => wry::http::Response::builder().status(500).body(error.to_string().into_bytes().into()).unwrap(),
            }
        }
        _ => wry::http::Response::builder().status(404).body(b"Resource not found".to_vec().into()).unwrap(),
    }
}
