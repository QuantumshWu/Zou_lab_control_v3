//! The operator's file dialogs: the Common Item Dialog (IFileOpenDialog / IFileSaveDialog), as
//! v2's QFileDialog shows it. Each opens where its kind says (SetFolder, [`dialog_start`]) with
//! v2's caption and filter; Windows' remembered folder of this executable never decides it.
use std::path::Path;

use windows::Win32::Foundation::HWND;
use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree, IServiceProvider};
use windows::Win32::UI::Shell::Common::COMDLG_FILTERSPEC;
use windows::Win32::UI::Shell::{
    FILEOPENDIALOGOPTIONS, FOS_FILEMUSTEXIST, FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST, FOS_PICKFOLDERS,
    FileOpenDialog, FileSaveDialog, IFileDialog, IFileOpenDialog, IFileSaveDialog, IShellItem, SHCreateItemFromParsingName, SIGDN_FILESYSPATH,
    FOS_FORCEPREVIEWPANEON, FOS_ALLOWMULTISELECT, FVM_THUMBNAIL, IFolderView, IFolderView2, IFileDialogEvents, IFileDialogEvents_Impl,
    FDE_OVERWRITE_RESPONSE, FDE_SHAREVIOLATION_RESPONSE, FDEOR_DEFAULT, FDESVR_DEFAULT,
};
use windows::core::{GUID, HSTRING, PCWSTR, Interface, Ref, implement};
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
    show(parent, title, Some(save), options, filter, start, home, false)
}

/// A visual selector for saved figures: archives and their image previews
/// appear together, with large thumbnails and the shell's preview pane.
pub fn figure(parent: isize, start: &Path, home: &Path) -> Result<Option<String>, String> {
    show(parent, "Open figure — choose data or its image preview", Some(false), FOS_FILEMUSTEXIST | FOS_FORCEPREVIEWPANEON,
        crate::figure_file::FILTER, start, home, true)
}

/// Ask for a folder (v2 FluentPathEdit mode="dir", QFileDialog.getExistingDirectory), opened at `start`, else in `home`.
pub fn folder(parent: isize, title: &str, start: &Path, home: &Path) -> Result<Option<String>, String> {
    show(parent, title, None, FOS_PICKFOLDERS, "", start, home, false)
}

#[implement(IFileDialogEvents)]
struct FigurePreview;

impl IFileDialogEvents_Impl for FigurePreview_Impl {
    fn OnFolderChange(&self, dialog: Ref<IFileDialog>) -> windows::core::Result<()> {
        // SID_SFolderView is IID_IFolderView. Query after the folder exists,
        // including when navigating into another folder inside the dialog.
        // A shell without this service keeps the preview pane and its normal
        // view controls; it must still allow the user to select an archive.
        if let Some(dialog) = dialog.as_ref() {
            unsafe {
                if let Ok(provider) = dialog.cast::<IServiceProvider>() {
                    if let Ok(view) = provider.QueryService::<IFolderView2>(&IFolderView::IID) {
                        let _ = view.SetViewModeAndIconSize(FVM_THUMBNAIL, 128);
                    }
                }
            }
        }
        Ok(())
    }
    fn OnFileOk(&self, _: Ref<IFileDialog>) -> windows::core::Result<()> { Ok(()) }
    fn OnFolderChanging(&self, _: Ref<IFileDialog>, _: Ref<IShellItem>) -> windows::core::Result<()> { Ok(()) }
    fn OnSelectionChange(&self, _: Ref<IFileDialog>) -> windows::core::Result<()> { Ok(()) }
    fn OnTypeChange(&self, _: Ref<IFileDialog>) -> windows::core::Result<()> { Ok(()) }
    fn OnShareViolation(&self, _: Ref<IFileDialog>, _: Ref<IShellItem>) -> windows::core::Result<FDE_SHAREVIOLATION_RESPONSE> { Ok(FDESVR_DEFAULT) }
    fn OnOverwrite(&self, _: Ref<IFileDialog>, _: Ref<IShellItem>) -> windows::core::Result<FDE_OVERWRITE_RESPONSE> { Ok(FDEOR_DEFAULT) }
}

/// `save`: Some(true) a save, Some(false) an open, None a folder.
fn show(parent: isize, title: &str, save: Option<bool>, options: FILEOPENDIALOGOPTIONS, filter: &str, start: &Path, home: &Path, previews: bool) -> Result<Option<String>, String> {
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
        let defaults = if previews { defaults & !FOS_ALLOWMULTISELECT } else { defaults };
        dialog.SetOptions(defaults | options | FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR | FOS_PATHMUSTEXIST).map_err(failed)?;
        // Keep the figure picker's remembered view separate from config/pulse
        // dialogs. Selection stays single-file; an NPZ can contain many datasets.
        let events = if previews {
            dialog.SetClientGuid(&GUID::from_u128(0x62fc78cf_546d_48d4_b251_441fc779ac70)).map_err(failed)?;
            let events: IFileDialogEvents = FigurePreview.into();
            let cookie = dialog.Advise(&events).map_err(failed)?;
            Some((events, cookie))
        } else { None };
        if !title.is_empty() { dialog.SetTitle(&HSTRING::from(title)).map_err(failed)?; }
        if !specs.is_empty() { dialog.SetFileTypes(&specs).map_err(failed)?; dialog.SetFileTypeIndex(1).map_err(failed)?; }
        if !extension.is_empty() { dialog.SetDefaultExtension(&HSTRING::from(extension)).map_err(failed)?; }
        let item: IShellItem = SHCreateItemFromParsingName(&HSTRING::from(folder.as_os_str()), None).map_err(failed)?;
        dialog.SetFolder(&item).map_err(failed)?;
        // A folder is never offered as a name: the dialog opens inside it.
        if save.is_some() && !name.is_empty() { dialog.SetFileName(&HSTRING::from(name)).map_err(failed)?; }
        let shown = dialog.Show(Some(HWND(parent as _)));
        if let Some((_, cookie)) = events { let _ = dialog.Unadvise(cookie); }
        if let Err(error) = shown {
            return if error.code().0 as u32 == CANCELLED { Ok(None) } else { Err(failed(error)) };
        }
        let chosen = dialog.GetResult().map_err(failed)?.GetDisplayName(SIGDN_FILESYSPATH).map_err(failed)?;
        let path = chosen.to_string().map_err(|error| error.to_string());
        CoTaskMemFree(Some(chosen.0 as *const _));
        path.map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figure_filter_shows_archives_and_previews_together() {
        let offered = types(crate::figure_file::FILTER);
        assert!(offered[0].1.contains("*.npz") && offered[0].1.contains("*.png") && offered[0].1.contains("*.jpg"));
        assert_eq!(offered[1].1, "*.npz");
        assert!(!offered[2].1.contains("*.npz"));
    }

    /// A hardware-free native UI smoke test. The supplied folder contains
    /// synthetic NPZ/image pairs; cancel or choose one to complete the test.
    #[test]
    #[ignore = "interactive native figure picker; requires ZLC_FIGURE_PICKER_FIXTURE"]
    fn native_figure_picker_smoke() {
        let fixture = std::env::var_os("ZLC_FIGURE_PICKER_FIXTURE").expect("fixture folder");
        let folder = Path::new(&fixture);
        assert!(folder.is_dir());
        let selected = figure(0, folder, folder).unwrap();
        if let Some(selected) = selected { assert!(crate::figure_file::archive_path(Path::new(&selected)).unwrap().is_file()); }
    }
}
