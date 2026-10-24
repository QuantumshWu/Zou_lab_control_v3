//! Hidden diagnostic of the real shared frame. No mouse movement, window
//! activation or minimize/maximize command is performed by this probe.
use serde_json::json;
use tao::{platform::windows::WindowExtWindows,window::Window};
use windows_sys::Win32::{Foundation::{POINT,RECT},Graphics::{Dwm::{DwmGetWindowAttribute,DWMWA_WINDOW_CORNER_PREFERENCE},Gdi::ClientToScreen},UI::{HiDpi::GetDpiForWindow,WindowsAndMessaging::*}};

pub fn inspect(window:&Window)->serde_json::Value{
    unsafe{
        let hwnd=window.hwnd() as windows_sys::Win32::Foundation::HWND;
        let class:Vec<u16>="ZlcFrameHit".encode_utf16().chain([0]).collect();
        let hit=FindWindowExW(hwnd,std::ptr::null_mut(),class.as_ptr(),std::ptr::null());
        let mut count=0;let mut child=std::ptr::null_mut();loop{child=FindWindowExW(hwnd,child,class.as_ptr(),std::ptr::null());if child.is_null(){break;}count+=1;}
        let mut rect:RECT=std::mem::zeroed();GetClientRect(hwnd,&mut rect);let width=rect.right;let height=rect.bottom;
        let dpi=GetDpiForWindow(hwnd).max(96);let qt=((dpi as f64/96.)+0.5).floor() as i32;let button=46*qt;
        let point=|x:i32,y:i32|{let mut p=POINT{x,y};ClientToScreen(hwnd,&mut p);((p.x as u16 as u32)|((p.y as u16 as u32)<<16))as isize};
        let at=|x:i32,y:i32|if hit.is_null(){-1}else{SendMessageW(hit,WM_NCHITTEST,0,point(x,y))};
        let hits=json!({"caption":at(100*qt,16*qt),"minimize":at(width-button*5/2,16*qt),"maximize":at(width-button*3/2,16*qt),"close":at(width-button/2,16*qt),"topLeft":at(1,1),"bottom":at(width/2,height-1)});
        if !hit.is_null(){SendMessageW(hit,WM_NCMOUSEMOVE,HTMAXBUTTON as usize,point(width-button*3/2,16*qt));}
        let hover=crate::frame::caption(window);
        if !hit.is_null(){SendMessageW(hit,WM_NCLBUTTONDOWN,HTMAXBUTTON as usize,point(width-button*3/2,16*qt));}
        let pressed=crate::frame::caption(window);
        if !hit.is_null(){SendMessageW(hit,WM_NCMOUSELEAVE,0,0);}
        let clear=crate::frame::caption(window);let mut corner=0i32;
        let corner_status=DwmGetWindowAttribute(hwnd,DWMWA_WINDOW_CORNER_PREFERENCE as u32,(&mut corner as *mut i32).cast(),4);
        let style=GetWindowLongPtrW(hwnd,GWL_STYLE)as u32;
        json!({"visible":window.is_visible(),"hitWindows":count,"clientWidth":width,"clientHeight":height,"dpi":dpi,"barDeviceHeight":crate::frame::bar_height(window),"captionStyle":style&WS_CAPTION==WS_CAPTION,"resizeStyle":style&WS_THICKFRAME!=0,"cornerPreference":if corner_status==0{Some(corner)}else{None},"hits":hits,"hover":hover,"pressed":pressed,"clear":clear})
    }
}
