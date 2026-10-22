//! The operator's file dialogs: the Common Item Dialog (IFileOpenDialog / IFileSaveDialog), as
//! v2's QFileDialog shows it. Each opens where its kind says (SetFolder, [`dialog_start`]) with
//! v2's caption and filter; Windows' remembered folder of this executable never decides it.
use std::path::Path;

use windows::Win32::Foundation::HWND;
use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree};
use windows::Win32::UI::Shell::Common::COMDLG_FILTERSPEC;
use windows::Win32::UI::Shell::{
    FILEOPENDIALOGOPTIONS, FOS_FILEMUSTEXIST, FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST, FOS_PICKFOLDERS,
    FileOpenDialog, FileSaveDialog, IFileDialog, IFileOpenDialog, IFileSaveDialog, IShellItem, SHCreateItemFromParsingName, SIGDN_FILESYSPATH,
};
use windows::core::{HSTRING, PCWSTR};
use zlc_workbench::workspace::dialog_start;

/// A v2 (Qt) filter, "Apparatus (*.json);;All files (*)", as the dialog's file types: each
/// entry's words and its patterns ("*.json", or "*.*" for Qt's "*"); none offers All files.
fn types(filter: &str) -> Vec<(String, String)> {
    let mut types: Vec<(String, String)> = filter.split(";;").map(str::trim).filter(|words| !words.is_empty()).map(|words| {
        let patterns = words.rsplit_once('(').map_or("", |(_, patterns)| patterns.trim_end_matches(')'));
        let patterns: Vec<&str> = patterns.split_whitespace().map(|pattern| if pattern == "*" { "*.*" } else { pattern }).collect();
        (words.to_owned(), patterns.join(";"))
    }).collect();
    if types.is_empty() { types.push(("All files (*)".into(), "*.*".into())); }
    types
}

/// Ask for a file to open or save: captioned `title`, offering `filter` (v2's), at `start` (a
/// folder, or a file shown in its folder), or in `home`, its kind's folder, when that is gone.
pub fn file(parent: isize, title: &str, filter: &str, save: bool, start: &Path, home: &Path) -> Result<Option<String>, String> {
    let options = if save { FOS_OVERWRITEPROMPT } else { FOS_FILEMUSTEXIST };
    show(parent, title, Some(save), options, filter, start, home)
}

/// Ask for a folder (v2 FluentPathEdit mode="dir", QFileDialog.getExistingDirectory), opened at `start`, else in `home`.
pub fn folder(parent: isize, title: &str, start: &Path, home: &Path) -> Result<Option<String>, String> {
    show(parent, title, None, FOS_PICKFOLDERS, "", start, home)
}

/// `save`: Some(true) a save, Some(false) an open, None a folder.
fn show(parent: isize, title: &str, save: Option<bool>, options: FILEOPENDIALOGOPTIONS, filter: &str, start: &Path, home: &Path) -> Result<Option<String>, String> {
    const CANCELLED: u32 = 0x8007_04C7; // HRESULT_FROM_WIN32(ERROR_CANCELLED)
    let failed = |error: windows::core::Error| format!("File dialog: {error}");
    let (folder, name) = dialog_start(start, home, save == Some(true));
    let types = if save.is_some() { types(filter) } else { Vec::new() };
    // A name typed bare takes the first type's extension (Qt suffixFromFilter).
    let extension = types.first().and_then(|(_, patterns)| patterns.split(';').next()).and_then(|pattern| pattern.strip_prefix("*."))
        .filter(|extension| !extension.contains(['*', '?'])).unwrap_or("");
    let words: Vec<(HSTRING, HSTRING)> = types.iter().map(|(words, patterns)| (HSTRING::from(words.as_str()), HSTRING::from(patterns.as_str()))).collect();
    let specs: Vec<COMDLG_FILTERSPEC> = words.iter().map(|(words, patterns)| COMDLG_FILTERSPEC { pszName: PCWSTR(words.as_ptr()), pszSpec: PCWSTR(patterns.as_ptr()) }).collect();
    // SAFETY: COM calls on the window's thread (already an STA for WebView2; a second
    // CoInitializeEx answers S_FALSE); `specs` points into `words`, alive until Show returns;
    // the returned string is freed once.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
        let dialog: IFileDialog = if save == Some(true) { CoCreateInstance::<_, IFileSaveDialog>(&FileSaveDialog, None, CLSCTX_INPROC_SERVER).map_err(failed)?.into() }
            else { CoCreateInstance::<_, IFileOpenDialog>(&FileOpenDialog, None, CLSCTX_INPROC_SERVER).map_err(failed)?.into() };
        let defaults = dialog.GetOptions().map_err(failed)?;
        dialog.SetOptions(defaults | options | FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR | FOS_PATHMUSTEXIST).map_err(failed)?;
        if !title.is_empty() { dialog.SetTitle(&HSTRING::from(title)).map_err(failed)?; }
        if !specs.is_empty() { dialog.SetFileTypes(&specs).map_err(failed)?; dialog.SetFileTypeIndex(1).map_err(failed)?; }
        if !extension.is_empty() { dialog.SetDefaultExtension(&HSTRING::from(extension)).map_err(failed)?; }
        let item: IShellItem = SHCreateItemFromParsingName(&HSTRING::from(folder.as_os_str()), None).map_err(failed)?;
        dialog.SetFolder(&item).map_err(failed)?;
        // A folder is never offered as a name: the dialog opens inside it.
        if save.is_some() && !name.is_empty() { dialog.SetFileName(&HSTRING::from(name)).map_err(failed)?; }
        if let Err(error) = dialog.Show(Some(HWND(parent as _))) {
            return if error.code().0 as u32 == CANCELLED { Ok(None) } else { Err(failed(error)) };
        }
        let chosen = dialog.GetResult().map_err(failed)?.GetDisplayName(SIGDN_FILESYSPATH).map_err(failed)?;
        let path = chosen.to_string().map_err(|error| error.to_string());
        CoTaskMemFree(Some(chosen.0 as *const _));
        path.map(Some)
    }
}
