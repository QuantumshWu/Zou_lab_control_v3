// The engine release archive's unpacker (tools\fetch_engine.ps1; tools\release.ps1 checks each
// archive with it before recording it): a tar in an xz stream, as tools\release_archive.py writes
// it. Compiled by Windows PowerShell's Add-Type (.NET Framework 4, C# 5), so a lab PC needs
// nothing beyond Windows itself: neither Windows 10's tar.exe nor .NET has an xz decoder.
//
// It decodes the xz container (its CRC32s, index and footer, each block's check -- CRC32, CRC64 or
// SHA-256 -- over the decoded data) with the filters xz writes for executables: LZMA2, optionally
// after the x86 branch converter. The tar holds regular files (and directories) under relative
// paths; anything else is refused, as is any path leaving the folder.
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

public static class ZlcTarXz
{
    // Unpacks the .tar.xz archive into the folder (made when missing).
    public static void Extract(string archive, string folder)
    {
        ExtractTar(Decode(File.ReadAllBytes(archive)), folder);
    }

    static Exception Bad(string what) { return new InvalidDataException("The archive is damaged or not a release archive: " + what + "."); }

    // ---- xz ----------------------------------------------------------------------------------

    // The data of one xz stream (the whole file).
    public static byte[] Decode(byte[] xz)
    {
        byte[] magic = { 0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00 };
        if (xz.Length < 32 || xz.Length % 4 != 0) throw Bad("not an xz stream");
        for (int i = 0; i < magic.Length; i++) if (xz[i] != magic[i]) throw Bad("no xz header");
        if (Crc32(xz, 6, 2) != U32(xz, 8)) throw Bad("xz header CRC32");
        if (xz[6] != 0 || xz[7] > 0x0F) throw Bad("xz stream flags");
        int check = xz[7];
        int checkSize = check == 0 ? 0 : 4 << ((check - 1) / 3);
        if (check != 0 && check != 1 && check != 4 && check != 10) throw Bad("xz check type " + check);

        int footer = xz.Length - 12;
        if (xz[footer + 10] != 0x59 || xz[footer + 11] != 0x5A) throw Bad("no xz footer (one xz stream, no padding)");
        if (Crc32(xz, footer + 4, 6) != U32(xz, footer)) throw Bad("xz footer CRC32");
        if (xz[footer + 8] != xz[6] || xz[footer + 9] != xz[7]) throw Bad("xz footer flags");
        long indexSize = ((long)U32(xz, footer + 4) + 1) * 4;
        if (indexSize > footer - 12) throw Bad("xz index size");
        int index = footer - (int)indexSize;

        // The index: every block's unpadded and uncompressed size.
        int p = index;
        if (xz[p++] != 0) throw Bad("xz index indicator");
        long records = Vli(xz, ref p, footer - 4);
        if (records > (footer - index) / 2) throw Bad("xz index record count");
        long[] unpadded = new long[records], sizes = new long[records];
        long total = 0;
        for (long r = 0; r < records; r++)
        {
            unpadded[r] = Vli(xz, ref p, footer - 4);
            sizes[r] = Vli(xz, ref p, footer - 4);
            total += sizes[r];
            if (total > int.MaxValue) throw Bad("more data than one unpacking holds");
        }
        while ((p - index) % 4 != 0) if (xz[p++] != 0) throw Bad("xz index padding");
        if (p != footer - 4 || Crc32(xz, index, p - index) != U32(xz, p)) throw Bad("xz index CRC32");

        byte[] output = new byte[total];
        int at = 12, written = 0;
        for (long r = 0; r < records; r++)
        {
            int block = at;
            if (xz[at] == 0) throw Bad("fewer xz blocks than the index names");
            int headerSize = (xz[at] + 1) * 4;
            if (at + headerSize > index) throw Bad("xz block header size");
            int headerEnd = at + headerSize - 4;
            if (Crc32(xz, at, headerSize - 4) != U32(xz, headerEnd)) throw Bad("xz block header CRC32");
            int flags = xz[at + 1];
            if ((flags & 0x3C) != 0) throw Bad("xz block flags");
            int h = at + 2;
            long compressed = (flags & 0x40) != 0 ? Vli(xz, ref h, headerEnd) : -1;
            long uncompressed = (flags & 0x80) != 0 ? Vli(xz, ref h, headerEnd) : -1;
            int filters = (flags & 3) + 1;
            bool x86 = false;
            uint x86Start = 0;
            long dictionary = -1;
            for (int f = 0; f < filters; f++)
            {
                long id = Vli(xz, ref h, headerEnd);
                long properties = Vli(xz, ref h, headerEnd);
                if (properties > headerEnd - h) throw Bad("xz filter properties");
                if (id == 0x21 && f == filters - 1 && properties == 1) dictionary = Lzma2.DictionarySize(xz[h]);
                else if (id == 0x04 && f == 0 && filters == 2 && (properties == 0 || properties == 4))
                {
                    x86 = true;
                    if (properties == 4) x86Start = U32(xz, h);
                }
                else throw Bad("xz filter " + id + " (this unpacker decodes LZMA2, after x86 at most)");
                h += (int)properties;
            }
            while (h < headerEnd) if (xz[h++] != 0) throw Bad("xz block header padding");
            if (uncompressed >= 0 && uncompressed != sizes[r]) throw Bad("xz block size against its index");

            at = headerEnd + 4;
            int end = written + (int)sizes[r];
            int data = at;
            at = new Lzma2(xz, at, index, output, written, end, dictionary).Run();
            long dataSize = at - data;
            if (compressed >= 0 && compressed != dataSize) throw Bad("xz block compressed size");
            while ((at - block) % 4 != 0) if (at >= index || xz[at++] != 0) throw Bad("xz block padding");
            if (x86) X86(output, written, end - written, x86Start);
            if (at + checkSize > index || !CheckMatches(check, output, written, end - written, xz, at)) throw Bad("xz block check");
            at += checkSize;
            if (unpadded[r] != headerSize + dataSize + checkSize) throw Bad("xz block against its index");
            written = end;
        }
        if (at != index) throw Bad("more xz blocks than the index names");
        return output;
    }

    static uint U32(byte[] b, int i) { return (uint)(b[i] | b[i + 1] << 8 | b[i + 2] << 16 | b[i + 3] << 24); }

    // xz's variable-length integer (at most 9 bytes, no redundant trailing zero byte).
    static long Vli(byte[] b, ref int p, int end)
    {
        long value = 0;
        for (int i = 0; i < 9; i++)
        {
            if (p >= end) throw Bad("xz integer");
            int x = b[p++];
            value |= (long)(x & 0x7F) << (7 * i);
            if ((x & 0x80) == 0)
            {
                if (x == 0 && i > 0) throw Bad("xz integer");
                return value;
            }
        }
        throw Bad("xz integer");
    }

    static bool CheckMatches(int check, byte[] data, int offset, int count, byte[] xz, int at)
    {
        if (check == 1) return Crc32(data, offset, count) == U32(xz, at);
        if (check == 4)
        {
            ulong crc = Crc64(data, offset, count);
            return (uint)crc == U32(xz, at) && (uint)(crc >> 32) == U32(xz, at + 4);
        }
        if (check == 10)
        {
            byte[] hash;
            using (SHA256 sha = SHA256.Create()) hash = sha.ComputeHash(data, offset, count);
            for (int i = 0; i < 32; i++) if (hash[i] != xz[at + i]) return false;
        }
        return true;
    }

    static uint[] crc32Table;
    static ulong[] crc64Table;

    static uint Crc32(byte[] data, int offset, int count)
    {
        if (crc32Table == null)
        {
            uint[] table = new uint[256];
            for (uint n = 0; n < 256; n++)
            {
                uint c = n;
                for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                table[n] = c;
            }
            crc32Table = table;
        }
        uint crc = 0xFFFFFFFFu;
        for (int i = offset; i < offset + count; i++) crc = crc32Table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        return ~crc;
    }

    static ulong Crc64(byte[] data, int offset, int count)
    {
        if (crc64Table == null)
        {
            ulong[] table = new ulong[256];
            for (ulong n = 0; n < 256; n++)
            {
                ulong c = n;
                for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xC96C5795D7870F42ul ^ (c >> 1) : c >> 1;
                table[n] = c;
            }
            crc64Table = table;
        }
        ulong crc = ulong.MaxValue;
        for (int i = offset; i < offset + count; i++) crc = crc64Table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        return ~crc;
    }

    // xz's x86 branch converter, decoding: a CALL or JMP's absolute target back to the relative
    // one it was (liblzma simple/x86.c), over one block's data from its start offset. The last
    // four bytes are never converted, as the encoder left them.
    static void X86(byte[] buffer, int offset, int size, uint start)
    {
        bool[] allowed = { true, true, true, false, true, false, false, false };
        int[] bitNumber = { 0, 1, 2, 2, 3, 3, 3, 3 };
        if (size < 5) return;
        uint prevMask = 0;
        uint prevPos = start - 5;
        int limit = size - 5;
        int i = 0;
        while (i <= limit)
        {
            int b = buffer[offset + i];
            if (b != 0xE8 && b != 0xE9) { i++; continue; }
            uint position = start + (uint)i;
            uint gap = position - prevPos;
            prevPos = position;
            if (gap > 5) prevMask = 0;
            else for (uint k = 0; k < gap; k++) { prevMask &= 0x77; prevMask <<= 1; }
            b = buffer[offset + i + 4];
            if ((b == 0 || b == 0xFF) && allowed[(prevMask >> 1) & 7] && (prevMask >> 1) < 0x10)
            {
                uint source = (uint)b << 24 | (uint)buffer[offset + i + 3] << 16 | (uint)buffer[offset + i + 2] << 8 | buffer[offset + i + 1];
                uint destination;
                while (true)
                {
                    destination = source - (position + 5);
                    if (prevMask == 0) break;
                    int bit = bitNumber[prevMask >> 1];
                    int top = (int)(destination >> (24 - bit * 8)) & 0xFF;
                    if (top != 0 && top != 0xFF) break;
                    source = destination ^ ((1u << (32 - bit * 8)) - 1);
                }
                buffer[offset + i + 4] = (byte)~(((destination >> 24) & 1) - 1);
                buffer[offset + i + 3] = (byte)(destination >> 16);
                buffer[offset + i + 2] = (byte)(destination >> 8);
                buffer[offset + i + 1] = (byte)destination;
                i += 5;
                prevMask = 0;
            }
            else
            {
                i++;
                prevMask |= 1;
                if (b == 0 || b == 0xFF) prevMask |= 0x10;
            }
        }
    }

    // ---- LZMA2 -------------------------------------------------------------------------------

    // One block's LZMA2 data decoded into output[start, end): its chunks, uncompressed or LZMA,
    // the dictionary being the output itself since the last dictionary reset.
    sealed class Lzma2
    {
        const int States = 12, PositionStates = 16, EndPositionModel = 14, FullDistances = 128, AlignBits = 4;

        readonly byte[] input, output;
        int inPos, inEnd;
        readonly int limit, end;
        int outPos, dictionaryStart;
        readonly long dictionary;
        uint range, code;
        int lc, lp, pb;
        int state;
        uint rep0, rep1, rep2, rep3;
        readonly ushort[] isMatch = new ushort[States * PositionStates], isRep = new ushort[States], isRepG0 = new ushort[States],
            isRepG1 = new ushort[States], isRepG2 = new ushort[States], isRep0Long = new ushort[States * PositionStates],
            positionSlot = new ushort[4 << 6], positionSpecial = new ushort[1 + FullDistances - EndPositionModel],
            align = new ushort[1 << AlignBits], literal = new ushort[0x300 << 4];
        readonly Length matchLength = new Length(), repLength = new Length();

        public Lzma2(byte[] input, int at, int limit, byte[] output, int start, int end, long dictionary)
        {
            this.input = input; inPos = at; this.limit = limit;
            this.output = output; outPos = start; this.end = end;
            this.dictionary = dictionary;
        }

        public static long DictionarySize(int properties)
        {
            if (properties > 40) throw Bad("LZMA2 dictionary size");
            if (properties == 40) return uint.MaxValue;
            return (long)(2 | (properties & 1)) << (properties / 2 + 11);
        }

        // Decodes every chunk up to the end marker; the position after it.
        public int Run()
        {
            bool needDictionaryReset = true, needProperties = true;
            while (true)
            {
                if (inPos >= limit) throw Bad("LZMA2 data ends early");
                int control = input[inPos++];
                if (control == 0) break;
                if (control >= 0xE0 || control == 1)
                {
                    needProperties = true;
                    needDictionaryReset = false;
                    dictionaryStart = outPos;
                }
                else if (needDictionaryReset) throw Bad("LZMA2 without a dictionary reset");
                if (control >= 0x80)
                {
                    if (limit - inPos < 4) throw Bad("LZMA2 chunk header");
                    int unpacked = ((control & 0x1F) << 16) + (input[inPos] << 8) + input[inPos + 1] + 1;
                    int packed = (input[inPos + 2] << 8) + input[inPos + 3] + 1;
                    inPos += 4;
                    if (control >= 0xC0)
                    {
                        if (inPos >= limit) throw Bad("LZMA2 properties");
                        SetProperties(input[inPos++]);
                        needProperties = false;
                        Reset();
                    }
                    else if (needProperties) throw Bad("LZMA2 chunk without properties");
                    else if (control >= 0xA0) Reset();
                    if (unpacked > end - outPos || packed > limit - inPos) throw Bad("LZMA2 chunk size");
                    Chunk(inPos + packed, outPos + unpacked);
                }
                else
                {
                    if (control > 2) throw Bad("LZMA2 control byte");
                    if (limit - inPos < 2) throw Bad("LZMA2 chunk header");
                    int size = (input[inPos] << 8) + input[inPos + 1] + 1;
                    inPos += 2;
                    if (size > end - outPos || size > limit - inPos) throw Bad("LZMA2 chunk size");
                    Buffer.BlockCopy(input, inPos, output, outPos, size);
                    inPos += size;
                    outPos += size;
                }
            }
            if (outPos != end) throw Bad("LZMA2 data size against the index");
            return inPos;
        }

        void SetProperties(int properties)
        {
            if (properties >= 9 * 5 * 5) throw Bad("LZMA properties");
            lc = properties % 9;
            lp = properties / 9 % 5;
            pb = properties / 45;
            if (lc + lp > 4) throw Bad("LZMA properties");
        }

        void Reset()
        {
            foreach (ushort[] probabilities in new[] { isMatch, isRep, isRepG0, isRepG1, isRepG2, isRep0Long, positionSlot, positionSpecial, align, literal })
                for (int i = 0; i < probabilities.Length; i++) probabilities[i] = 1024;
            matchLength.Reset();
            repLength.Reset();
            state = 0;
            rep0 = rep1 = rep2 = rep3 = 0;
        }

        // One LZMA chunk: its own range coder over input[inPos, packedEnd), decoding up to outEnd.
        void Chunk(int packedEnd, int outEnd)
        {
            inEnd = packedEnd;
            if (inEnd - inPos < 5 || input[inPos] != 0) throw Bad("LZMA range coder start");
            code = (uint)(input[inPos + 1] << 24 | input[inPos + 2] << 16 | input[inPos + 3] << 8 | input[inPos + 4]);
            range = uint.MaxValue;
            inPos += 5;
            int positionMask = (1 << pb) - 1;
            while (outPos < outEnd)
            {
                int filled = outPos - dictionaryStart;
                int positionState = filled & positionMask;
                if (Bit(isMatch, (state << 4) + positionState) == 0) { Literal(filled); continue; }
                uint length;
                if (Bit(isRep, state) == 0)
                {
                    rep3 = rep2; rep2 = rep1; rep1 = rep0;
                    length = matchLength.Decode(this, positionState);
                    state = state < 7 ? 7 : 10;
                    rep0 = Distance(length);
                }
                else
                {
                    if (filled == 0) throw Bad("LZMA repeat before any data");
                    if (Bit(isRepG0, state) == 0)
                    {
                        if (Bit(isRep0Long, (state << 4) + positionState) == 0)
                        {
                            state = state < 7 ? 9 : 11;
                            Copy(1, outEnd);
                            continue;
                        }
                    }
                    else
                    {
                        uint distance;
                        if (Bit(isRepG1, state) == 0) distance = rep1;
                        else
                        {
                            if (Bit(isRepG2, state) == 0) distance = rep2;
                            else { distance = rep3; rep3 = rep2; }
                            rep2 = rep1;
                        }
                        rep1 = rep0;
                        rep0 = distance;
                    }
                    length = repLength.Decode(this, positionState);
                    state = state < 7 ? 8 : 11;
                }
                Copy((int)length + 2, outEnd);
            }
            if (inPos != inEnd || code != 0) throw Bad("LZMA chunk end");
            inPos = inEnd;
        }

        void Literal(int filled)
        {
            int previous = filled > 0 ? output[outPos - 1] : 0;
            int at = 0x300 * ((((filled & ((1 << lp) - 1)) << lc)) + (previous >> (8 - lc)));
            int symbol = 1;
            if (state >= 7)
            {
                if (rep0 >= filled) throw Bad("LZMA distance");
                int match = output[outPos - (int)rep0 - 1];
                while (symbol < 0x100)
                {
                    int matchBit = (match >> 7) & 1;
                    match <<= 1;
                    int bit = Bit(literal, at + ((1 + matchBit) << 8) + symbol);
                    symbol = (symbol << 1) | bit;
                    if (matchBit != bit) break;
                }
            }
            while (symbol < 0x100) symbol = (symbol << 1) | Bit(literal, at + symbol);
            output[outPos++] = (byte)symbol;
            state = state < 4 ? 0 : state < 10 ? state - 3 : state - 6;
        }

        uint Distance(uint length)
        {
            int slot = Tree(positionSlot, (int)Math.Min(length, 3) << 6, 6);
            if (slot < 4) return (uint)slot;
            int direct = (slot >> 1) - 1;
            uint distance = (uint)(2 | (slot & 1)) << direct;
            if (slot < EndPositionModel) return distance + Reverse(positionSpecial, (int)distance - slot, direct);
            for (int i = 0; i < direct - AlignBits; i++)
            {
                range >>= 1;
                code -= range;
                uint t = 0 - (code >> 31);
                code += range & t;
                distance += (t + 1) << (direct - 1 - i);
                Normalize();
            }
            return distance + Reverse(align, 0, AlignBits);
        }

        // Repeats `length` bytes from rep0 back; LZMA2 never has the end marker (distance 2^32 - 1).
        void Copy(int length, int outEnd)
        {
            if (rep0 >= outPos - dictionaryStart || rep0 >= dictionary) throw Bad("LZMA distance");
            if (length > outEnd - outPos) throw Bad("LZMA match past its chunk");
            int from = outPos - (int)rep0 - 1;
            if (rep0 + 1 >= length) Buffer.BlockCopy(output, from, output, outPos, length);
            else for (int i = 0; i < length; i++) output[outPos + i] = output[from + i];
            outPos += length;
        }

        int Bit(ushort[] probabilities, int i)
        {
            uint probability = probabilities[i];
            uint bound = (range >> 11) * probability;
            int bit;
            if (code < bound)
            {
                range = bound;
                probabilities[i] = (ushort)(probability + ((2048 - probability) >> 5));
                bit = 0;
            }
            else
            {
                range -= bound;
                code -= bound;
                probabilities[i] = (ushort)(probability - (probability >> 5));
                bit = 1;
            }
            Normalize();
            return bit;
        }

        void Normalize()
        {
            if (range >= 1u << 24) return;
            if (inPos >= inEnd) throw Bad("LZMA chunk ends early");
            range <<= 8;
            code = code << 8 | input[inPos++];
        }

        int Tree(ushort[] probabilities, int at, int bits)
        {
            int m = 1;
            for (int i = 0; i < bits; i++) m = (m << 1) | Bit(probabilities, at + m);
            return m - (1 << bits);
        }

        uint Reverse(ushort[] probabilities, int at, int bits)
        {
            int m = 1;
            uint symbol = 0;
            for (int i = 0; i < bits; i++)
            {
                int bit = Bit(probabilities, at + m);
                m = (m << 1) | bit;
                symbol |= (uint)bit << i;
            }
            return symbol;
        }

        // A match length less two: 3 bits (low), 3 more (mid) or 8 (high).
        sealed class Length
        {
            readonly ushort[] choice = new ushort[2], low = new ushort[PositionStates << 3], mid = new ushort[PositionStates << 3], high = new ushort[256];

            public void Reset()
            {
                foreach (ushort[] probabilities in new[] { choice, low, mid, high })
                    for (int i = 0; i < probabilities.Length; i++) probabilities[i] = 1024;
            }

            public uint Decode(Lzma2 decoder, int positionState)
            {
                if (decoder.Bit(choice, 0) == 0) return (uint)decoder.Tree(low, positionState << 3, 3);
                if (decoder.Bit(choice, 1) == 0) return 8 + (uint)decoder.Tree(mid, positionState << 3, 3);
                return 16 + (uint)decoder.Tree(high, 0, 8);
            }
        }
    }

    // ---- tar ---------------------------------------------------------------------------------

    // The tar's regular files and directories written under the folder (a ustar or v7 header per
    // entry, the data padded to 512 bytes, two zero blocks at the end), each file's time kept.
    static void ExtractTar(byte[] tar, string folder)
    {
        string root = Path.GetFullPath(folder).TrimEnd('\\') + "\\";
        Directory.CreateDirectory(root);
        int at = 0;
        while (true)
        {
            if (tar.Length - at < 512) throw Bad("the tar ends without its end blocks");
            if (Zero(tar, at, 512)) return;
            long sum = 0;
            for (int i = 0; i < 512; i++) sum += i >= 148 && i < 156 ? 0x20 : tar[at + i];
            if (Octal(tar, at + 148, 8) != sum) throw Bad("tar header checksum");
            string name = Text(tar, at, 100);
            if (Text(tar, at + 257, 5) == "ustar")
            {
                string prefix = Text(tar, at + 345, 155);
                if (prefix.Length > 0) name = prefix + "/" + name;
            }
            long size = Octal(tar, at + 124, 12);
            char type = (char)tar[at + 156];
            int data = at + 512;
            if (size > tar.Length - data) throw Bad("tar entry size");
            string path = Target(root, name);
            if (type == '5') Directory.CreateDirectory(path);
            else if (type == '0' || type == '\0')
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                using (FileStream file = new FileStream(path, FileMode.CreateNew, FileAccess.Write)) file.Write(tar, data, (int)size);
                File.SetLastWriteTimeUtc(path, new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddSeconds(Octal(tar, at + 136, 12)));
            }
            else throw Bad("tar entry " + name + " of type '" + type + "' (only files and folders)");
            at = data + (int)((size + 511) / 512 * 512);
        }
    }

    static bool Zero(byte[] b, int at, int count)
    {
        for (int i = at; i < at + count; i++) if (b[i] != 0) return false;
        return true;
    }

    static string Text(byte[] b, int at, int count)
    {
        int length = 0;
        while (length < count && b[at + length] != 0) length++;
        return Encoding.UTF8.GetString(b, at, length);
    }

    static long Octal(byte[] b, int at, int count)
    {
        long value = 0;
        int i = 0;
        while (i < count && b[at + i] == ' ') i++;
        int digits = 0;
        for (; i < count && b[at + i] >= '0' && b[at + i] <= '7'; i++, digits++) value = value * 8 + (b[at + i] - '0');
        for (; i < count; i++) if (b[at + i] != 0 && b[at + i] != ' ') throw Bad("tar number");
        if (digits == 0) throw Bad("tar number");
        return value;
    }

    // Where an entry's relative path lands: inside the folder, never through '..', a drive or a root.
    static string Target(string root, string name)
    {
        string[] parts = name.TrimEnd('/').Split('/');
        foreach (string part in parts)
            if (part.Length == 0 || part == "." || part == ".." || part.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
                throw Bad("tar path " + name);
        string path = Path.GetFullPath(root + string.Join("\\", parts));
        if (!path.StartsWith(root, StringComparison.OrdinalIgnoreCase)) throw Bad("tar path " + name);
        return path;
    }
}
