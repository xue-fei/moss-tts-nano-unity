using System;
using System.Collections.Generic;

namespace MossTtsNano
{
    /// <summary>
    /// SentencePiece 归一化器，对应 sentencepiece/src/normalizer.cc 的
    /// Normalizer::Normalize / Normalizer::NormalizePrefix。
    ///
    /// 本模型的 normalizer_spec.name = "nmt_nfkc"，规则全部编码在
    /// precompiled_charsmap（237561 字节）里，格式为：
    ///   [trie_size:uint32][DoubleArray trie][normalized string pool]
    /// trie 命中后拿到的 value 是 pool 里的偏移，规则替换串以 '\0' 结尾。
    ///
    /// 之前 C# 侧只做了「空格 → ▁ + 句首补 ▁」，缺了这一整套规则，
    /// 导致全角逗号「，」这类字符没有折叠成半角，走 byte-fallback
    /// 变成 &lt;0xEF&gt;&lt;0xBC&gt;&lt;0x8C&gt; 三个无语义 token，
    /// 直接表现为合成音频里的停顿和怪异语气音。
    ///
    /// 输出是 UTF-8 字节串（不是 string）：后续 BPE 合并完全在字节层面进行，
    /// 且 byte-fallback 也要按字节切分，转成 string 会丢失这个粒度。
    /// </summary>
    internal sealed class SpNormalizer
    {
        /// <summary>U+2581 LOWER ONE EIGHTH BLOCK 的 UTF-8 编码。</summary>
        private static readonly byte[] SpaceSymbol = { 0xE2, 0x96, 0x81 };
        private static readonly byte[] AsciiSpace = { 0x20 };

        private readonly SpDoubleArray _trie;
        private readonly byte[] _normalized;   // 规则替换串池（含 '\0' 分隔）
        private readonly int _normalizedStart;
        private readonly bool _addDummyPrefix;
        private readonly bool _removeExtraWhitespaces;
        private readonly byte[] _space;
        private readonly SpPrefixMatcher _matcher;

        public SpNormalizer(
            byte[] charsMap,
            bool addDummyPrefix,
            bool removeExtraWhitespaces,
            bool escapeWhitespaces,
            SpPrefixMatcher matcher)
        {
            if (charsMap == null || charsMap.Length <= 4)
                throw new ArgumentException("precompiled_charsmap is empty or broken", nameof(charsMap));

            // DecodePrecompiledCharsMap：头 4 字节是 little-endian 的 trie 字节数
            int trieSize = charsMap[0] | (charsMap[1] << 8) | (charsMap[2] << 16) | (charsMap[3] << 24);
            if (trieSize < 1024 || (trieSize & 0x3FF) != 0)
                throw new ArgumentException($"trie size {trieSize} is not a positive multiple of 1024");
            if (trieSize >= charsMap.Length - 4)
                throw new ArgumentException("trie size exceeds charsmap length");

            _trie = new SpDoubleArray(charsMap, 4, trieSize);
            _normalized = charsMap;
            _normalizedStart = 4 + trieSize;
            _addDummyPrefix = addDummyPrefix;
            _removeExtraWhitespaces = removeExtraWhitespaces;
            _space = escapeWhitespaces ? SpaceSymbol : AsciiSpace;
            _matcher = matcher;
        }

        /// <summary>
        /// string_util::OneCharLen —— 由 UTF-8 首字节推断序列长度。
        /// 非法的续接字节（0x80..0xBF）按 1 处理，与原实现一致。
        /// </summary>
        internal static int OneCharLen(byte b)
        {
            if (b < 0xC0) return 1;
            if (b < 0xE0) return 2;
            if (b < 0xF0) return 3;
            return 4;
        }

        /// <summary>
        /// Normalizer::NormalizePrefix —— 取输入 pos 处最长命中的替换规则。
        /// 命中不到规则时原样输出一个完整 UTF-8 字符。
        /// </summary>
        /// <param name="repStart">替换串在 buffer 中的起始下标</param>
        /// <param name="repLength">替换串字节数</param>
        /// <param name="repBuffer">替换串所在的数组（规则池或输入本身）</param>
        /// <returns>本次消耗的输入字节数</returns>
        private int NormalizePrefix(
            byte[] input, int pos, int end,
            out byte[] repBuffer, out int repStart, out int repLength)
        {
            // user_defined 符号（<|im_start|> 等）必须原样保留，不参与归一化。
            if (_matcher != null && _matcher.PrefixMatch(input, pos, end, out int matchLen))
            {
                repBuffer = input;
                repStart = pos;
                repLength = matchLen;
                return matchLen;
            }

            if (_trie.LongestPrefix(input, pos, end, out uint value, out int length) &&
                length <= end - pos && _normalizedStart + (int)value < _normalized.Length)
            {
                int start = _normalizedStart + (int)value;
                int stop = start;
                while (stop < _normalized.Length && _normalized[stop] != 0) stop++;
                repBuffer = _normalized;
                repStart = start;
                repLength = stop - start;
                return length;
            }

            int charLen = Math.Min(OneCharLen(input[pos]), end - pos);
            repBuffer = input;
            repStart = pos;
            repLength = charLen;
            return charLen;
        }

        /// <summary>
        /// Normalizer::Normalize —— 返回归一化后的 UTF-8 字节串。
        /// </summary>
        public byte[] Normalize(byte[] input)
        {
            if (input == null || input.Length == 0) return Array.Empty<byte>();

            int pos = 0;
            int end = input.Length;

            // 跳过开头的空白（规则可能把 \t、\u3000、\u00A0 等都映射成 " "）
            if (_removeExtraWhitespaces)
            {
                while (pos < end)
                {
                    int consumed = NormalizePrefix(input, pos, end, out byte[] buf, out int s, out int len);
                    if (!(len == 1 && buf[s] == 0x20)) break;
                    pos += consumed;
                }
            }

            if (pos >= end) return Array.Empty<byte>();

            var output = new List<byte>(input.Length + input.Length / 2);

            // add_dummy_prefix：句首补一个 ▁，让 "world" 与 "hello world" 都以 ▁ 开头
            if (_addDummyPrefix) output.AddRange(_space);

            bool isPrevSpace = _removeExtraWhitespaces;
            while (pos < end)
            {
                int consumed = NormalizePrefix(input, pos, end, out byte[] buf, out int s, out int len);

                // 上一个片段以空白结尾时，吃掉本片段开头的连续空白
                while (isPrevSpace && len > 0 && buf[s] == 0x20)
                {
                    s++;
                    len--;
                }

                if (len > 0)
                {
                    for (int i = 0; i < len; i++)
                    {
                        byte b = buf[s + i];
                        if (b == 0x20) output.AddRange(_space);
                        else output.Add(b);
                    }
                    isPrevSpace = buf[s + len - 1] == 0x20;
                }

                pos += consumed;
                if (!_removeExtraWhitespaces) isPrevSpace = false;
            }

            // 去掉结尾的 ▁
            if (_removeExtraWhitespaces)
            {
                while (EndsWithSpace(output))
                    output.RemoveRange(output.Count - _space.Length, _space.Length);
            }

            return output.ToArray();
        }

        private bool EndsWithSpace(List<byte> output)
        {
            if (output.Count < _space.Length) return false;
            int offset = output.Count - _space.Length;
            for (int i = 0; i < _space.Length; i++)
                if (output[offset + i] != _space[i]) return false;
            return true;
        }
    }
}
