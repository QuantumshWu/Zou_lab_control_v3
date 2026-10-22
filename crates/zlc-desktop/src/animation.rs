//! A run's animation (`run.animation`, PLUGIN_PROTOCOL §5): its frame
//! previews as one looping GIF, each frame on its own palette, as v2's
//! Pillow save of the recording's PNGs (`duration`, `loop=0`, `disposal=2`).
use std::collections::HashMap;
use std::io::{BufReader, Write};
use std::path::Path;

/// One PNG as opaque RGB over white.
fn decode(path: &Path) -> Result<(u32, u32, Vec<[u8; 3]>), String> {
    let file = std::fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|error| format!("{}: {error}", path.display()))?;
    let mut buffer = vec![0; reader.output_buffer_size().ok_or("the image is too large")?];
    let info = reader.next_frame(&mut buffer).map_err(|error| format!("{}: {error}", path.display()))?;
    let over_white = |value: u8, alpha: u8| ((u16::from(value) * u16::from(alpha) + 255 * (255 - u16::from(alpha)) + 127) / 255) as u8;
    let pixels = buffer[..info.buffer_size()].chunks_exact(info.color_type.samples()).map(|pixel| match pixel {
        [gray] => [*gray; 3],
        [gray, alpha] => [over_white(*gray, *alpha); 3],
        [red, green, blue] => [*red, *green, *blue],
        [red, green, blue, alpha] => [over_white(*red, *alpha), over_white(*green, *alpha), over_white(*blue, *alpha)],
        _ => [255; 3],
    }).collect();
    Ok((info.width, info.height, pixels))
}

/// At most 256 colours and each pixel's index: the exact colours when they
/// fit, else median cut (each box's weighted mean).
fn quantize(pixels: &[[u8; 3]]) -> (Vec<[u8; 3]>, Vec<u8>) {
    let mut counts: HashMap<[u8; 3], u64> = HashMap::new();
    for pixel in pixels { *counts.entry(*pixel).or_default() += 1; }
    let colors: Vec<([u8; 3], u64)> = counts.into_iter().collect();
    let mut boxes: Vec<Vec<usize>> = vec![(0..colors.len()).collect()];
    while boxes.len() < 256 {
        let range = |members: &[usize], channel: usize| {
            let values = members.iter().map(|&index| colors[index].0[channel]);
            values.clone().max().unwrap_or(0) - values.min().unwrap_or(0)
        };
        let Some((widest, channel)) = boxes.iter().enumerate().filter(|(_, members)| members.len() > 1)
            .flat_map(|(index, members)| (0..3).map(move |channel| (index, channel, range(members, channel))))
            .max_by_key(|(_, _, spread)| *spread).filter(|(_, _, spread)| *spread > 0).map(|(index, channel, _)| (index, channel)) else { break };
        let mut members = boxes.swap_remove(widest);
        members.sort_unstable_by_key(|&index| colors[index].0[channel]);
        let total: u64 = members.iter().map(|&index| colors[index].1).sum();
        let (mut seen, mut split) = (0, 1);
        for (position, &index) in members.iter().enumerate() {
            seen += colors[index].1;
            if seen * 2 >= total { split = (position + 1).clamp(1, members.len() - 1); break; }
        }
        let upper = members.split_off(split);
        boxes.push(members);
        boxes.push(upper);
    }
    let mut palette = Vec::with_capacity(boxes.len());
    let mut slot: HashMap<[u8; 3], u8> = HashMap::with_capacity(colors.len());
    for (index, members) in boxes.iter().enumerate() {
        let weight: u64 = members.iter().map(|&member| colors[member].1).sum();
        let mean = |channel: usize| ((members.iter().map(|&member| u64::from(colors[member].0[channel]) * colors[member].1).sum::<u64>() + weight / 2) / weight) as u8;
        palette.push([mean(0), mean(1), mean(2)]);
        for &member in members { slot.insert(colors[member].0, index as u8); }
    }
    (palette, pixels.iter().map(|pixel| slot[pixel]).collect())
}

/// GIF's variable-width LZW of 8-bit indices.
fn lzw(indices: &[u8]) -> Vec<u8> {
    const CLEAR: usize = 256;
    const END: usize = 257;
    let mut table = vec![0u16; 4096 * 256];
    let (mut output, mut bits, mut filled) = (Vec::new(), 0u64, 0u32);
    let mut emit = |code: usize, size: u32, output: &mut Vec<u8>| {
        bits |= (code as u64) << filled;
        filled += size;
        while filled >= 8 { output.push(bits as u8); bits >>= 8; filled -= 8; }
    };
    let (mut next, mut size) = (END + 1, 9);
    emit(CLEAR, size, &mut output);
    let Some((&first, rest)) = indices.split_first() else {
        emit(END, size, &mut output);
        if filled > 0 { output.push(bits as u8); }
        return output;
    };
    let mut prefix = usize::from(first);
    for &symbol in rest {
        let entry = prefix * 256 + usize::from(symbol);
        if table[entry] != 0 { prefix = usize::from(table[entry]); continue; }
        emit(prefix, size, &mut output);
        if next < 4096 {
            table[entry] = next as u16;
            next += 1;
            if next > 1 << size && size < 12 { size += 1; }
        } else {
            emit(CLEAR, size, &mut output);
            table.fill(0);
            (next, size) = (END + 1, 9);
        }
        prefix = usize::from(symbol);
    }
    emit(prefix, size, &mut output);
    emit(END, size, &mut output);
    if filled > 0 { output.push(bits as u8); }
    output
}

/// The GIF of these frames, `frame_ms` each, looping.
pub fn gif(frames: &[&Path], frame_ms: u32) -> Result<Vec<u8>, String> {
    let decoded = frames.iter().map(|path| decode(path)).collect::<Result<Vec<_>, _>>()?;
    let width = decoded.iter().map(|frame| frame.0).max().ok_or("an animation needs frames")?;
    let height = decoded.iter().map(|frame| frame.1).max().unwrap_or(0);
    let (width, height) = (u16::try_from(width).map_err(|_| "a frame is too wide")?, u16::try_from(height).map_err(|_| "a frame is too tall")?);
    let mut out = b"GIF89a".to_vec();
    out.extend(width.to_le_bytes());
    out.extend(height.to_le_bytes());
    out.extend([0x70, 0, 0]); // no global colour table, 8 bits of colour resolution
    out.extend(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00"); // loop forever
    let delay = u16::try_from(frame_ms.div_ceil(10)).unwrap_or(u16::MAX).to_le_bytes();
    for (frame_width, frame_height, pixels) in decoded {
        out.extend([0x21, 0xF9, 4, 2 << 2, delay[0], delay[1], 0, 0]); // restore to background after the frame
        out.push(0x2C);
        out.extend([0, 0, 0, 0]);
        out.extend((frame_width as u16).to_le_bytes());
        out.extend((frame_height as u16).to_le_bytes());
        out.push(0x80 | 7); // a local 256-colour table
        let (palette, indices) = quantize(&pixels);
        out.extend(palette.iter().flatten());
        out.extend(std::iter::repeat_n(0, 3 * (256 - palette.len())));
        out.push(8);
        for block in lzw(&indices).chunks(255) {
            out.push(block.len() as u8);
            out.write_all(block).map_err(|error| error.to_string())?;
        }
        out.push(0);
    }
    out.push(0x3B);
    Ok(out)
}

#[cfg(test)]
mod tests {
    /// The LZW stream decodes back to its indices (a reference GIF decoder).
    #[test]
    fn lzw_round_trips() {
        let decode = |data: &[u8]| {
            let (mut bits, mut filled, mut position) = (0u64, 0u32, 0usize);
            let mut read = |size: u32| {
                while filled < size { bits |= u64::from(data[position]) << filled; position += 1; filled += 8; }
                let code = (bits & ((1 << size) - 1)) as usize;
                bits >>= size;
                filled -= size;
                code
            };
            let (mut table, mut size, mut output, mut previous): (Vec<Vec<u8>>, u32, Vec<u8>, Option<usize>) = (Vec::new(), 9, Vec::new(), None);
            loop {
                let code = read(size);
                if code == 256 { table = (0..=255).map(|value| vec![value as u8]).chain([Vec::new(), Vec::new()]).collect(); size = 9; previous = None; continue; }
                if code == 257 { return output; }
                let entry = match (code < table.len(), previous) {
                    (true, _) => table[code].clone(),
                    (false, Some(previous)) => { let mut entry = table[previous].clone(); entry.push(table[previous][0]); entry }
                    (false, None) => panic!("bad code"),
                };
                if let Some(previous) = previous {
                    if table.len() < 4096 {
                        let mut added = table[previous].clone();
                        added.push(entry[0]);
                        table.push(added);
                        if table.len() == 1 << size && size < 12 { size += 1; }
                    }
                }
                output.extend(&entry);
                previous = Some(code);
            }
        };
        let mut state = 7u32;
        let noise: Vec<u8> = (0..200_000).map(|_| { state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345); (state >> 16) as u8 }).collect();
        let runs: Vec<u8> = (0..300_000).map(|index| (index / 97 % 7) as u8).collect();
        for indices in [noise, runs, vec![5], Vec::new()] { assert_eq!(decode(&super::lzw(&indices)), indices); }
    }
}
