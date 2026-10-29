//! Saved figures are an NPZ archive and, usually, a same-named image.
//! Images are visual handles for that exact sibling archive, never a fuzzy
//! filename search (two runs can share a prefix or live in different folders).
use std::path::{Path, PathBuf};

pub const FILTER: &str = "Figures and previews (*.npz *.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp);;Figure archives (*.npz);;Image previews (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)";

pub fn archive_path(selected: &Path) -> Result<PathBuf, String> {
    if !selected.is_file() { return Err(format!("Figure file not found: {}", selected.display())); }
    let extension = selected.extension().and_then(|value| value.to_str()).unwrap_or("").to_ascii_lowercase();
    if extension == "npz" { return Ok(selected.to_path_buf()); }
    if !["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff", "webp"].contains(&extension.as_str()) {
        return Err("Choose an .npz figure archive or its PNG, JPEG, BMP, GIF, TIFF or WebP preview.".into());
    }
    let archive = selected.with_extension("npz");
    if archive.is_file() { return Ok(archive); }
    // Support upper-case archive extensions in case-sensitive directories,
    // too. Keep the complete stem and directory exact, and refuse ambiguity.
    let folder = selected.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let matches: Vec<PathBuf> = std::fs::read_dir(folder).map_err(|error| format!("Cannot read {}: {error}", folder.display()))?
        .filter_map(Result::ok).map(|entry| entry.path())
        .filter(|path| path.file_stem() == selected.file_stem()
            && path.extension().and_then(|value| value.to_str()).is_some_and(|extension| extension.eq_ignore_ascii_case("npz"))
            && path.is_file()).collect();
    match matches.as_slice() {
        [archive] => Ok(archive.clone()),
        [] => Err(format!("No matching data archive for {}. Expected {} in the same folder. Choose the .npz file directly if it has a different name.",
            selected.display(), archive.display())),
        _ => Err(format!("More than one matching .npz archive for {}. Choose the intended .npz file directly.", selected.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Folder(PathBuf);
    impl Folder {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("zlc-figure-choice-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn file(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, b"path resolution fixture").unwrap();
            path
        }
    }
    impl Drop for Folder { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

    #[test]
    fn archive_and_preview_choose_the_exact_same_stem_in_the_same_folder() {
        let first = Folder::new();
        let other = Folder::new();
        let archive = first.file("扫描 01.v2.npz");
        first.file("扫描 01.v2-extra.npz");
        let other_archive = other.file("扫描 01.v2.npz");
        assert_eq!(archive_path(&archive).unwrap(), archive);
        for extension in ["png", "JPG", "jpeg", "bmp", "gif", "tif", "tiff", "webp"] {
            let preview = first.file(&format!("扫描 01.v2.{extension}"));
            assert_eq!(archive_path(&preview).unwrap(), archive);
            assert_eq!(archive_path(&other.file(&format!("扫描 01.v2.{extension}"))).unwrap(), other_archive);
        }
    }

    #[test]
    fn missing_pair_never_opens_a_prefix_match_or_another_folders_archive() {
        let folder = Folder::new();
        let other = Folder::new();
        let preview = folder.file("scan.png");
        folder.file("scan-extra.npz");
        other.file("scan.npz");
        let error = archive_path(&preview).unwrap_err();
        assert!(error.contains("No matching data archive") && error.contains("scan.npz"));
        assert!(archive_path(&folder.0.join("missing.npz")).unwrap_err().contains("not found"));
        assert!(archive_path(&folder.0).unwrap_err().contains("not found"));
        assert!(archive_path(&folder.file("scan.txt")).unwrap_err().contains("Choose an .npz"));
    }

    #[test]
    fn extension_case_does_not_change_the_pair() {
        let folder = Folder::new();
        let archive = folder.file("sample.NPZ");
        assert_eq!(archive_path(&archive).unwrap(), archive);
        let resolved = archive_path(&folder.file("sample.PNG")).unwrap();
        assert_eq!(std::fs::canonicalize(resolved).unwrap(), std::fs::canonicalize(archive).unwrap());
    }
}
