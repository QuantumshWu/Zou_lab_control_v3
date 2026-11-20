//! The operator's file dialogs: the Common Item Dialog (IFileOpenDialog / IFileSaveDialog), as
//! v2's QFileDialog shows it. Each opens where its kind says (SetFolder, [`dialog_start`]) with
//! v2's caption and filter; Windows' remembered folder of this executable never decides it.
//!
//! A command from the operator that needs a path and names none asks for it first ([`Dialogs::ask`]):
//! the dialog runs on a thread of its own, owned by the asking window (which Windows disables while
//! it is up), so the event loop and every other window go on. Its answer comes back as an event;
//! [`Dialogs::answered`] gives the command again with the chosen path in it, to run as if the
//! operator had named it, or the reply a cancelled (or failed) dialog makes to its request.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use windows::Win32::Foundation::HWND;
use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, IServiceProvider};
use windows::Win32::UI::Shell::Common::COMDLG_FILTERSPEC;
use windows::Win32::UI::Shell::{
    FILEOPENDIALOGOPTIONS, FOS_FILEMUSTEXIST, FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST, FOS_PICKFOLDERS,
    FileOpenDialog, FileSaveDialog, IFileDialog, IFileOpenDialog, IFileSaveDialog, IShellItem, SHCreateItemFromParsingName, SIGDN_FILESYSPATH,
    FOS_FORCEPREVIEWPANEON, FOS_ALLOWMULTISELECT, FVM_THUMBNAIL, IFolderView, IFolderView2, IFileDialogEvents, IFileDialogEvents_Impl,
    FDE_OVERWRITE_RESPONSE, FDE_SHAREVIOLATION_RESPONSE, FDEOR_DEFAULT, FDESVR_DEFAULT,
};
use windows::core::{GUID, HSTRING, PCWSTR, Interface, Ref, implement};
use zlc_workbench::workspace::dialog_start;

use crate::windows::Kind;

/// What a dialog chooses: a file to open or to save, offering v2's filter; a folder; a saved figure.
pub enum Choose { Open(String), Save(String), Folder, Figure }

/// A dialog: what it chooses, its caption, where it opens (`start`: a folder, or a file shown in
/// its folder) and its kind's folder (`home`), where it opens when `start` is gone.
pub struct Spec { pub choose: Choose, pub title: String, pub start: PathBuf, pub home: PathBuf }

impl Spec {
    /// A file to open (`save` false) or to save, as the workbench's and the Pulse Editor's `file_dialog` give it.
    pub fn file(title: &str, filter: &str, save: bool, start: PathBuf, home: PathBuf) -> Self {
        let choose = if save { Choose::Save(filter.to_owned()) } else { Choose::Open(filter.to_owned()) };
        Spec { choose, title: title.to_owned(), start, home }
    }
}

/// A command waiting for the path its dialog asks: the window that gave it, the command, where in
/// it the path goes (a JSON pointer), what else an answer sets (a save dialog that asked before
/// replacing a file says so), and the sections a reply without a path also carries.
pub struct Ask { window: Option<Kind>, command: Value, at: &'static str, marks: Vec<(&'static str, Value)>, unanswered: Value }

impl Ask {
    /// `command`, given in `window`, takes its path as `path`.
    pub fn new(window: Option<Kind>, command: Value) -> Self { Ask { window, command, at: "/path", marks: Vec::new(), unanswered: json!({}) } }
    /// The path goes at `pointer` instead.
    pub fn at(mut self, pointer: &'static str) -> Self { self.at = pointer; self }
    /// An answer also sets `value` at `pointer`.
    pub fn marking(mut self, pointer: &'static str, value: Value) -> Self { self.marks.push((pointer, value)); self }
    /// A reply without a path also carries these sections.
    pub fn unanswered(mut self, sections: Value) -> Self { self.unanswered = sections; self }

    /// The command as if the operator had named `path`.
    fn answered(mut self, path: String) -> Value {
        put(&mut self.command, self.at, json!(path));
        for (pointer, value) in std::mem::take(&mut self.marks) { put(&mut self.command, pointer, value); }
        self.command
    }

    /// The reply to the command's request when no path came: a null result, with the error when the dialog failed.
    fn reply(&self, error: Option<String>) -> Value {
        let mut reply = json!({"type": "state", "request": self.command["id"], "error": error, "result": null});
        if let (Some(reply), Some(sections)) = (reply.as_object_mut(), self.unanswered.as_object()) {
            for (section, value) in sections { reply.insert(section.clone(), value.clone()); }
        }
        reply
    }
}

/// Set `value` at the JSON `pointer` in `target`, making the objects on the way.
fn put(target: &mut Value, pointer: &str, value: Value) {
    let mut at = target;
    for key in pointer.split('/').skip(1) {
        if !at.is_object() { *at = json!({}); }
        at = at.as_object_mut().expect("an object").entry(key.replace("~1", "/").replace("~0", "~")).or_insert(Value::Null);
    }
    *at = value;
}

/// What a dialog's answer leads to: its command run with the path, or a reply to its window.
pub enum Answered { Run { window: Option<Kind>, command: Value }, Reply { window: Option<Kind>, reply: Value } }

/// A dialog's answer: the path chosen, None when cancelled, or why it could not be shown.
pub type Answer = Result<Option<String>, String>;

/// The dialogs open now, one per window at most; `post` takes each answer back to the event loop.
pub struct Dialogs<P> { open: BTreeSet<Option<Kind>>, post: P }

impl<P: Fn(Ask, Answer) + Clone + Send + 'static> Dialogs<P> {
    pub fn new(post: P) -> Self { Dialogs { open: BTreeSet::new(), post } }

    /// Show `spec` for `ask`, owned by `parent` (the asking window), on a thread of its own; its
    /// answer is posted. Only the operator at a window is asked (v2 present=False): a client
    /// (`operator` false) or a run names its paths. A window whose dialog is open is refused another.
    pub fn ask(&mut self, parent: isize, spec: Spec, ask: Ask, operator: bool) -> Result<(), String> {
        if !operator || ask.command.get("_run").is_some() {
            return Err(format!("{} needs its path named: no dialog asks a client or a run", ask.command["action"].as_str().unwrap_or("this command")));
        }
        let window = ask.window.clone();
        self.claim(&window)?;
        let post = self.post.clone();
        std::thread::Builder::new().name("zlc-file-dialog".into()).spawn(move || {
            // SAFETY: this thread's own apartment, single-threaded as the dialog needs, ended once it answered.
            let initialized = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) }.is_ok();
            let answer = show(parent, &spec);
            if initialized { unsafe { CoUninitialize() }; }
            post(ask, answer);
        }).map(|_| ()).map_err(|error| { self.open.remove(&window); format!("File dialog: {error}") })
    }

    fn claim(&mut self, window: &Option<Kind>) -> Result<(), String> {
        if self.open.insert(window.clone()) { Ok(()) } else { Err("This window's file dialog is open: answer it first".into()) }
    }

    /// `ask`'s dialog answered: its window may ask again.
    pub fn answered(&mut self, ask: Ask, answer: Answer) -> Answered {
        self.open.remove(&ask.window);
        match answer {
            Ok(Some(path)) => Answered::Run { window: ask.window.clone(), command: ask.answered(path) },
            Ok(None) => Answered::Reply { reply: ask.reply(None), window: ask.window },
            Err(error) => Answered::Reply { reply: ask.reply(Some(error)), window: ask.window },
        }
    }
}

/// Show `spec`, owned by `parent`, on this thread (a COM single-threaded apartment): the chosen path, None when cancelled.
fn show(parent: isize, spec: &Spec) -> Answer {
    let Spec { choose, title, start, home } = spec;
    match choose {
        Choose::Open(filter) => show_dialog(parent, title, Some(false), FOS_FILEMUSTEXIST, filter, start, home, false),
        Choose::Save(filter) => show_dialog(parent, title, Some(true), FOS_OVERWRITEPROMPT, filter, start, home, false),
        Choose::Folder => show_dialog(parent, title, None, FOS_PICKFOLDERS, "", start, home, false),
        // A visual selector for saved figures: archives and their image previews
        // appear together, with large thumbnails and the shell's preview pane.
        Choose::Figure => show_dialog(parent, title, Some(false), FOS_FILEMUSTEXIST | FOS_FORCEPREVIEWPANEON, crate::figure_file::FILTER, start, home, true),
    }
}

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
fn show_dialog(parent: isize, title: &str, save: Option<bool>, options: FILEOPENDIALOGOPTIONS, filter: &str, start: &Path, home: &Path, previews: bool) -> Result<Option<String>, String> {
    const CANCELLED: u32 = 0x8007_04C7; // HRESULT_FROM_WIN32(ERROR_CANCELLED)
    let failed = |error: windows::core::Error| format!("File dialog: {error}");
    let (folder, name) = dialog_start(start, home, save == Some(true));
    let types = if save.is_some() { types(filter) } else { Vec::new() };
    // A name typed bare takes the first type's extension (Qt suffixFromFilter).
    let extension = types.first().and_then(|(_, patterns)| patterns.split(';').next()).and_then(|pattern| pattern.strip_prefix("*."))
        .filter(|extension| !extension.contains(['*', '?'])).unwrap_or("");
    let words: Vec<(HSTRING, HSTRING)> = types.iter().map(|(words, patterns)| (HSTRING::from(words.as_str()), HSTRING::from(patterns.as_str()))).collect();
    let specs: Vec<COMDLG_FILTERSPEC> = words.iter().map(|(words, patterns)| COMDLG_FILTERSPEC { pszName: PCWSTR(words.as_ptr()), pszSpec: PCWSTR(patterns.as_ptr()) }).collect();
    // SAFETY: COM calls on the dialog's thread, a single-threaded apartment ([`Dialogs::ask`]);
    // `specs` points into `words`, alive until Show returns; the returned string is freed once.
    unsafe {
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

    fn nowhere() -> Dialogs<impl Fn(Ask, Answer) + Clone + Send + 'static> { Dialogs::new(|_: Ask, _: Answer| {}) }

    #[test]
    fn a_chosen_path_runs_the_command_as_if_named() {
        let mut dialogs = nowhere();
        let ask = Ask::new(Some(Kind::Devices), json!({"id": 7, "action": "save_apparatus"})).marking("/replace_unread", json!(true));
        dialogs.claim(&ask.window).unwrap();
        let Answered::Run { window, command } = dialogs.answered(ask, Ok(Some("C:/a.json".into()))) else { panic!("a chosen path runs") };
        assert_eq!(window, Some(Kind::Devices));
        assert_eq!(command, json!({"id": 7, "action": "save_apparatus", "path": "C:/a.json", "replace_unread": true}));
        // A path inside the command (a data editor's Save as intent) goes where its site says.
        let ask = Ask::new(Some(Kind::FigureViewer(2)), json!({"id": 8, "action": "data_editor", "editor": "e1", "intent": {"op": "save_as", "note": "n"}}))
            .at("/intent/path").marking("/intent/asked", json!(true));
        let Answered::Run { command, .. } = dialogs.answered(ask, Ok(Some("C:/f.npz".into()))) else { panic!("a chosen path runs") };
        assert_eq!(command["intent"], json!({"op": "save_as", "note": "n", "path": "C:/f.npz", "asked": true}));
        assert_eq!(command["editor"], "e1");
    }

    #[test]
    fn a_window_has_one_dialog_at_a_time_and_others_go_on() {
        let mut dialogs = nowhere();
        let pulse = Some(Kind::PulseEditor);
        dialogs.claim(&pulse).unwrap();
        assert!(dialogs.claim(&pulse).is_err(), "a second ask of the same window is refused");
        dialogs.claim(&Some(Kind::TaskConsole)).unwrap();
        dialogs.answered(Ask::new(pulse.clone(), json!({"id": 1, "action": "pulse_open"})), Ok(None));
        dialogs.claim(&pulse).unwrap();
    }

    #[test]
    fn no_dialog_asks_a_client_or_a_run() {
        let mut dialogs = nowhere();
        let spec = || Spec::file("Save apparatus", "Apparatus (*.json)", true, PathBuf::new(), PathBuf::new());
        let refused = dialogs.ask(0, spec(), Ask::new(None, json!({"id": 3, "action": "save_apparatus"})), false).unwrap_err();
        assert!(refused.starts_with("save_apparatus needs its path named"), "{refused}");
        assert!(dialogs.ask(0, spec(), Ask::new(Some(Kind::TaskConsole), json!({"id": 4, "action": "save_apparatus", "_run": "r1"})), true).is_err());
        // Neither refusal holds the window.
        dialogs.claim(&None).unwrap();
        dialogs.claim(&Some(Kind::TaskConsole)).unwrap();
    }

    #[test]
    fn a_cancel_replies_null_to_its_request() {
        let mut dialogs = nowhere();
        let ask = Ask::new(Some(Kind::TaskConsole), json!({"id": 9, "action": "choose_file", "title": "Pick"}));
        dialogs.claim(&ask.window).unwrap();
        let Answered::Reply { window, reply } = dialogs.answered(ask, Ok(None)) else { panic!("a cancel replies") };
        assert_eq!(window, Some(Kind::TaskConsole));
        assert_eq!(reply, json!({"type": "state", "request": 9, "error": null, "result": null}));
        // A site's own sections go with it (a data editor's Save as: nothing saved), and a failed dialog says why.
        let ask = Ask::new(Some(Kind::FigureViewer(1)), json!({"id": 10, "action": "data_editor"})).unanswered(json!({"saved": null}));
        let Answered::Reply { reply, .. } = dialogs.answered(ask, Err("File dialog: no".into())) else { panic!("a failure replies") };
        assert_eq!(reply, json!({"type": "state", "request": 10, "error": "File dialog: no", "result": null, "saved": null}));
    }

    /// A hardware-free native UI smoke test of the whole ask: the supplied folder contains
    /// synthetic NPZ/image pairs; cancel or choose one to complete the test.
    #[test]
    #[ignore = "interactive native figure picker; requires ZLC_FIGURE_PICKER_FIXTURE"]
    fn native_figure_picker_smoke() {
        let fixture = std::env::var_os("ZLC_FIGURE_PICKER_FIXTURE").expect("fixture folder");
        let folder = PathBuf::from(&fixture);
        assert!(folder.is_dir());
        let (sender, answers) = std::sync::mpsc::channel();
        let mut dialogs = Dialogs::new(move |ask: Ask, answer: Answer| { let _ = sender.send((ask, answer)); });
        let spec = Spec { choose: Choose::Figure, title: "Open figure".into(), start: folder.clone(), home: folder };
        dialogs.ask(0, spec, Ask::new(None, json!({"id": 1, "action": "open"})), true).unwrap();
        let (ask, answer) = answers.recv().unwrap();
        if let Answered::Run { command, .. } = dialogs.answered(ask, answer) {
            assert!(crate::figure_file::archive_path(Path::new(command["path"].as_str().unwrap())).unwrap().is_file());
        }
    }
}
