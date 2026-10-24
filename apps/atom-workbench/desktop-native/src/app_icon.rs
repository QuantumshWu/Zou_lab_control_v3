//! The exact v3 product icon, loaded from this executable's resource group 1.
use tao::window::{Icon, Window};
pub fn source_sha256()->String{use sha2::{Sha256,Digest};format!("{:x}",Sha256::digest(include_bytes!(env!("ATOM_V3_ICON_FILE"))))}

#[cfg(target_os="windows")]
pub fn icon() -> Option<Icon> {
    use tao::platform::windows::IconExtWindows;
    Icon::from_resource(1, None).ok()
}
#[cfg(not(target_os="windows"))]
pub fn icon() -> Option<Icon> { None }

/// Hidden integration checks can verify the actual native large/small icon
/// handles without opening or activating a visible user window.
#[cfg(target_os="windows")]
pub fn window_icons_available(window:&Window) -> (bool,bool) {
    use tao::platform::windows::WindowExtWindows;
    use windows_sys::Win32::UI::WindowsAndMessaging::{SendMessageW,WM_GETICON,ICON_BIG,ICON_SMALL};
    let hwnd=window.hwnd() as _;
    // The Window owns HWND; both messages synchronously read its icon handles.
    unsafe {
        (SendMessageW(hwnd,WM_GETICON,ICON_BIG as usize,0)!=0,
         SendMessageW(hwnd,WM_GETICON,ICON_SMALL as usize,0)!=0)
    }
}
#[cfg(not(target_os="windows"))]
pub fn window_icons_available(_window:&Window) -> (bool,bool) { (false,false) }
