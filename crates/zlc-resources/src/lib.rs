//! The Windows resources every executable of the project carries: the product
//! icon (zlc-desktop's assets/zlc.ico) as RT_ICON images and RT_GROUP_ICON 1,
//! which Explorer, the taskbar and shortcuts show, and an application manifest
//! where the package has one. A package with a binary takes this crate as its
//! build-dependency and calls [`embed`] from its build script, which writes a
//! compiled resource file (.res) the linker takes as it is -- no rc.exe, no mt.exe.
use std::path::PathBuf;

/// The product icon; zlc-desktop also serves it to the page's title bar.
const ICON: &[u8] = include_bytes!("../../zlc-desktop/assets/zlc.ico");

/// Gives every binary of the calling package the product icon and, when given,
/// `manifest` (RT_MANIFEST 1). Called from a build script; nothing for a non-MSVC target.
pub fn embed(manifest: Option<&[u8]>) {
    embed_icon(ICON,manifest);
}

/// Share the Windows resource writer with independently packaged v3 tools.
/// The existing entry keeps the normal v3 product icon.
pub fn embed_icon(icon:&[u8],manifest:Option<&[u8]>) {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    let mut resources = Resources::new();
    if let Some(manifest) = manifest {
        resources.entry(24, 1, 0x1030, 0x0409, manifest);
    }
    resources.icon(icon);
    let path = PathBuf::from(std::env::var("OUT_DIR").expect("a build script's OUT_DIR")).join("resources.res");
    std::fs::write(&path, &resources.0).expect("resources.res");
    println!("cargo:rustc-link-arg-bins={}", path.display());
}

/// A .res: the empty entry every one starts with, then each resource as a
/// 32-byte header and its DWORD-padded data.
struct Resources(Vec<u8>);

impl Resources {
    fn new() -> Resources {
        let mut resources = Resources(Vec::new());
        resources.entry(0, 0, 0, 0, &[]);
        resources
    }

    fn entry(&mut self, kind: u16, name: u16, flags: u16, language: u16, data: &[u8]) {
        let out = &mut self.0;
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(32u32.to_le_bytes());
        for word in [0xffff, kind, 0xffff, name] {
            out.extend(word.to_le_bytes());
        }
        out.extend(0u32.to_le_bytes());
        out.extend(flags.to_le_bytes());
        out.extend(language.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(data);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }

    /// An .ico file as the executable's icon: each image an RT_ICON (ids 1..),
    /// then the RT_GROUP_ICON 1 listing them.
    fn icon(&mut self, ico: &[u8]) {
        let count = u16::from_le_bytes([ico[4], ico[5]]);
        let mut group = vec![0, 0, 1, 0];
        group.extend(count.to_le_bytes());
        for index in 0..count {
            let entry = &ico[6 + 16 * usize::from(index)..22 + 16 * usize::from(index)];
            let size = u32::from_le_bytes(entry[8..12].try_into().unwrap()) as usize;
            let offset = u32::from_le_bytes(entry[12..16].try_into().unwrap()) as usize;
            self.entry(3, index + 1, 0x1010, 0x0409, &ico[offset..offset + size]);
            group.extend(&entry[0..4]); // width, height, colours, reserved
            group.extend(1u16.to_le_bytes()); // planes
            group.extend(&entry[6..12]); // bit count, bytes in the image
            group.extend((index + 1).to_le_bytes());
        }
        self.entry(14, 1, 0x1030, 0x0409, &group);
    }
}
