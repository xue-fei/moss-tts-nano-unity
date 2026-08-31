using System;
using System.Collections.Generic;

namespace MossTtsNano
{
    /// <summary>
    /// 对应 sentencepiece/src/normalizer.cc 的 normalizer::PrefixMatcher。
    ///
    /// 由词表里所有 USER_DEFINED 类型的 piece（本模型 11 个：
    /// &lt;|im_start|&gt;、&lt;|audio_start|&gt;、&lt;user_inst&gt; 等）构成，
    /// 作用有两处：
    ///   1. 归一化时保护这些符号，不让 nmt_nfkc 规则改写它们；
    ///   2. BPE 切分时把它们作为一个不可再分（freeze）的整体符号。
    ///
    /// 原实现用 darts trie；符号只有十来个、且都以 '&lt;' 开头，
    /// 这里用「首字节索引 + 按长度倒序比较」即可，语义完全等价：
    /// 两者都只关心"最长命中长度"与"是否命中"。
    /// </summary>
    internal sealed class SpPrefixMatcher
    {
        private readonly Dictionary<byte, List<byte[]>> _byFirstByte;
        private readonly bool _empty;

        public SpPrefixMatcher(IEnumerable<string> symbols)
        {
            _byFirstByte = new Dictionary<byte, List<byte[]>>();

            foreach (string symbol in symbols)
            {
                if (string.IsNullOrEmpty(symbol)) continue;
                byte[] bytes = System.Text.Encoding.UTF8.GetBytes(symbol);
                if (bytes.Length == 0) continue;

                if (!_byFirstByte.TryGetValue(bytes[0], out var list))
                {
                    list = new List<byte[]>();
                    _byFirstByte[bytes[0]] = list;
                }
                list.Add(bytes);
            }

            // 长的排前面，第一个命中即最长命中
            foreach (var list in _byFirstByte.Values)
                list.Sort((a, b) => b.Length.CompareTo(a.Length));

            _empty = _byFirstByte.Count == 0;
        }

        /// <summary>
        /// PrefixMatcher::PrefixMatch —— 命中时输出最长命中的字节长度。
        /// 未命中时输出一个完整 UTF-8 字符的长度并返回 false，
        /// 与原实现的 `min(w.size(), OneCharLen(w.data()))` 一致。
        /// </summary>
        public bool PrefixMatch(byte[] buf, int pos, int end, out int matchLength)
        {
            if (!_empty && _byFirstByte.TryGetValue(buf[pos], out var candidates))
            {
                int available = end - pos;
                foreach (byte[] candidate in candidates)
                {
                    if (candidate.Length > available) continue;
                    bool hit = true;
                    for (int i = 0; i < candidate.Length; i++)
                    {
                        if (buf[pos + i] != candidate[i]) { hit = false; break; }
                    }
                    if (hit)
                    {
                        matchLength = candidate.Length;
                        return true;
                    }
                }
            }

            matchLength = Math.Min(SpNormalizer.OneCharLen(buf[pos]), end - pos);
            return false;
        }
    }
}
