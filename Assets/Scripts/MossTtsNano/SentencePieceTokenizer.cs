using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// SentencePiece 分词器（C# 实现），逐行对齐官方 C++ 实现：
    ///
    ///   1. Normalizer::Normalize        —— SpNormalizer（nmt_nfkc + ▁ 转义）
    ///   2. PrefixMatcher                —— SpPrefixMatcher（保护 user_defined）
    ///   3. bpe::Model::SampleEncode(α=0) —— 本文件的 EncodeBytes（BPE 合并）
    ///   4. byte_fallback 分解            —— 未登录片段拆成 &lt;0xNN&gt;
    ///
    /// 本模型是 **BPE**（trainer_spec.model_type = 2），不是 Unigram：
    /// scores 是 0, -1, -2, … 即取负的合并序号（rank），
    /// 所以不能用 Viterbi 最大得分路径，必须按 rank 从高到低做贪心合并。
    ///
    /// 旧实现用「贪心最长匹配 + 缺失归一化」，在
    ///   欢迎关注模思智能，这是一个语音合成测试。
    /// 上会把全角逗号拆成 3 个 byte-fallback token、并把「这是一个」错切成
    /// 「这是」+「一个」，模型收到偏离训练分布的 token 序列后产生停顿与怪音。
    /// 现实现已用官方 sentencepiece 在 3000+ 条随机语料上逐 token 验证一致。
    /// </summary>
    public class SentencePieceTokenizer : ITokenizer
    {
        private const int TypeNormal = 1;
        private const int TypeUnknown = 2;
        private const int TypeControl = 3;
        private const int TypeUserDefined = 4;
        private const int TypeUnused = 5;
        private const int TypeByte = 6;

        private readonly string[] _pieces;
        private readonly float[] _scores;
        private readonly int[] _types;
        private readonly Dictionary<string, int> _pieceToId;
        private readonly SpByteMap _mergeIds;      // 字节串 → id（已剔除 control/unknown）
        private readonly int[] _byteFallbackIds;   // 0..255 → <0xNN> 的 id
        private readonly int _maxPieceBytes;
        private readonly bool _byteFallback;
        private readonly SpNormalizer _normalizer;
        private readonly SpPrefixMatcher _matcher;

        private readonly int _unkId;
        private readonly int _bosId;
        private readonly int _eosId;
        private readonly int _padId;

        public int UnkId => _unkId;
        public int BosId => _bosId;
        public int EosId => _eosId;
        public int PadId => _padId;
        public int VocabSize => _pieces.Length;

        // ==== 复用的工作缓冲，避免每次 Encode 都重新分配 ====
        private int[] _symStart = new int[256];
        private int[] _symLength = new int[256];
        private bool[] _symFreeze = new bool[256];
        private int[] _symPrev = new int[256];
        private int[] _symNext = new int[256];
        private readonly SpSymbolPairHeap _agenda = new SpSymbolPairHeap(512);

        /// <summary>
        /// 从 export_tokenizer_assets.py 导出的 tokenizer_sp.json 加载。
        /// 归一化规则表（tokenizer_charsmap.bytes）默认取同目录同名文件，
        /// 文件名写在 json 的 charsmap_file 字段里。
        /// </summary>
        public SentencePieceTokenizer(string spJsonPath)
        {
            var data = JsonConvert.DeserializeObject<SpModelData>(File.ReadAllText(spJsonPath));
            if (data?.pieces == null || data.pieces.Length == 0)
                throw new InvalidDataException($"tokenizer json has no pieces: {spJsonPath}");
            if (data.scores == null || data.scores.Length != data.pieces.Length)
                throw new InvalidDataException("tokenizer json: scores length mismatch");
            if (data.types == null || data.types.Length != data.pieces.Length)
                throw new InvalidDataException("tokenizer json: types length mismatch");

            _pieces = data.pieces;
            _scores = data.scores;
            _types = data.types;
            _byteFallback = data.byte_fallback;
            _unkId = data.unk_id;
            _bosId = data.bos_id;
            _eosId = data.eos_id;
            _padId = data.pad_id;

            _pieceToId = new Dictionary<string, int>(_pieces.Length);
            _mergeIds = new SpByteMap(_pieces.Length);
            var userDefined = new List<string>();
            _maxPieceBytes = 0;

            for (int i = 0; i < _pieces.Length; i++)
            {
                string piece = _pieces[i];
                _pieceToId[piece] = i;

                if (_types[i] == TypeUserDefined) userDefined.Add(piece);

                // reserved_id_map_ 里的 CONTROL / UNKNOWN 不参与 BPE 合并，
                // 对应原实现的 PieceToIdNoReserved + IsReservedId 双重排除。
                if (_types[i] == TypeControl || _types[i] == TypeUnknown) continue;

                byte[] bytes = PieceToBytes(piece, _types[i]);
                if (bytes.Length == 0) continue;
                _mergeIds.Add(bytes, i);
                if (bytes.Length > _maxPieceBytes) _maxPieceBytes = bytes.Length;
            }

            _byteFallbackIds = new int[256];
            for (int b = 0; b < 256; b++)
            {
                _byteFallbackIds[b] = _pieceToId.TryGetValue($"<0x{b:X2}>", out int id) ? id : _unkId;
            }

            _matcher = new SpPrefixMatcher(userDefined);

            string charsMapName = string.IsNullOrEmpty(data.charsmap_file)
                ? "tokenizer_charsmap.bytes"
                : data.charsmap_file;
            string charsMapPath = Path.Combine(
                Path.GetDirectoryName(spJsonPath) ?? ".", charsMapName);
            if (!File.Exists(charsMapPath))
                throw new FileNotFoundException(
                    $"precompiled_charsmap not found: {charsMapPath}. " +
                    "Run MOSS-TTS-Nano/export_tokenizer_assets.py to regenerate tokenizer assets.",
                    charsMapPath);

            _normalizer = new SpNormalizer(
                File.ReadAllBytes(charsMapPath),
                data.add_dummy_prefix,
                data.remove_extra_whitespaces,
                data.escape_whitespaces,
                _matcher);

            Debug.Log(
                $"[SentencePiece] {_pieces.Length} pieces, normalizer={data.normalizer_name}, " +
                $"byteFallback={_byteFallback}, maxPieceBytes={_maxPieceBytes}, " +
                $"userDefined={userDefined.Count}");
        }

        /// <summary>
        /// BYTE 类型的 piece 字面量是 "&lt;0x8C&gt;"，但它在 BPE 合并里代表的是
        /// **那一个原始字节**。词表里的字面量拿去做字节匹配会匹配不到任何输入，
        /// 因此这里按类型还原成真实字节。
        /// </summary>
        private static byte[] PieceToBytes(string piece, int type)
        {
            if (type == TypeByte && piece.Length == 6 &&
                piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>')
            {
                int hi = HexValue(piece[3]);
                int lo = HexValue(piece[4]);
                if (hi >= 0 && lo >= 0) return new[] { (byte)((hi << 4) | lo) };
            }
            return System.Text.Encoding.UTF8.GetBytes(piece);
        }

        private static int HexValue(char c)
        {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            return -1;
        }

        public List<int> Encode(string text)
        {
            var ids = new List<int>();
            if (string.IsNullOrEmpty(text)) return ids;

            byte[] normalized = _normalizer.Normalize(System.Text.Encoding.UTF8.GetBytes(text));
            if (normalized.Length == 0) return ids;

            EncodeBytes(normalized, ids);
            return ids;
        }

        /// <summary>
        /// bpe::Model::SampleEncode 的 alpha = 0 分支（无 dropout）。
        ///
        /// 步骤：把输入切成初始符号（user_defined 整块且 freeze，其余按 UTF-8
        /// 单字符）→ 把所有相邻二元组按 score 入堆 → 反复取出 score 最高的
        /// 二元组做合并、并把新产生的左右二元组入堆 → 按链表顺序输出。
        /// </summary>
        private void EncodeBytes(byte[] buf, List<int> ids)
        {
            int n = 0;
            EnsureSymbolCapacity(buf.Length);

            for (int pos = 0; pos < buf.Length;)
            {
                bool found = _matcher.PrefixMatch(buf, pos, buf.Length, out int mblen);
                _symStart[n] = pos;
                _symLength[n] = mblen;
                _symFreeze[n] = found;
                _symPrev[n] = n - 1;
                _symNext[n] = pos + mblen < buf.Length ? n + 1 : -1;
                pos += mblen;
                n++;
            }
            if (n == 0) return;
            _symPrev[0] = -1;

            _agenda.Clear();
            for (int left = 0; left + 1 < n; left++)
                MaybeAddPair(buf, left, left + 1);

            while (_agenda.Count > 0)
            {
                SpSymbolPairHeap.Pair top = _agenda.Pop();
                int left = top.Left;
                int right = top.Right;

                // 陈旧条目：任一侧已被吞并，或两侧长度和已变化
                if (_symLength[left] == 0 || _symLength[right] == 0) continue;
                if (_symLength[left] + _symLength[right] != top.Size) continue;

                _symLength[left] = top.Size;
                _symLength[right] = 0;
                _symNext[left] = _symNext[right];
                if (_symNext[right] >= 0) _symPrev[_symNext[right]] = left;

                MaybeAddPair(buf, _symPrev[left], left);
                MaybeAddPair(buf, left, _symNext[left]);
            }

            for (int idx = 0; idx != -1; idx = _symNext[idx])
            {
                int length = _symLength[idx];
                if (length == 0) continue;
                int start = _symStart[idx];

                if (_mergeIds.TryGet(buf, start, length, out int id))
                {
                    ids.Add(id);
                    continue;
                }

                // 未登录片段：byte_fallback 时拆成 <0xNN>，否则退化为 <unk>
                if (_byteFallback)
                {
                    for (int i = 0; i < length; i++)
                        ids.Add(_byteFallbackIds[buf[start + i]]);
                }
                else
                {
                    ids.Add(_unkId);
                }
            }
        }

        private void MaybeAddPair(byte[] buf, int left, int right)
        {
            if (left == -1 || right == -1) return;
            if (_symFreeze[left] || _symFreeze[right]) return;
            if (_symLength[left] == 0 || _symLength[right] == 0) return;

            int size = _symLength[left] + _symLength[right];
            if (size > _maxPieceBytes) return;   // 不可能命中词表，省一次哈希
            if (!_mergeIds.TryGet(buf, _symStart[left], size, out int id)) return;
            if (_types[id] == TypeUnused) return;

            _agenda.Push(_scores[id], left, right, size);
        }

        private void EnsureSymbolCapacity(int capacity)
        {
            if (_symStart.Length >= capacity) return;
            int size = _symStart.Length;
            while (size < capacity) size <<= 1;
            _symStart = new int[size];
            _symLength = new int[size];
            _symFreeze = new bool[size];
            _symPrev = new int[size];
            _symNext = new int[size];
        }

        /// <summary>
        /// 解码：把 piece 还原成文本。▁ → 空格，&lt;0xNN&gt; 按字节收集后统一
        /// 按 UTF-8 解码（逐字节强转 char 会把中文变成乱码）。
        /// </summary>
        public string Decode(List<int> tokenIds)
        {
            if (tokenIds == null || tokenIds.Count == 0) return string.Empty;

            var bytes = new List<byte>(tokenIds.Count * 3);
            foreach (int id in tokenIds)
            {
                if (id < 0 || id >= _pieces.Length) continue;

                int type = _types[id];
                if (type == TypeControl || type == TypeUnknown) continue;

                if (type == TypeByte)
                {
                    bytes.AddRange(PieceToBytes(_pieces[id], type));
                    continue;
                }

                string piece = _pieces[id].Replace("\u2581", " ");
                bytes.AddRange(System.Text.Encoding.UTF8.GetBytes(piece));
            }

            // SentencePiece 解码会去掉 add_dummy_prefix 补出来的首个空格
            string text = System.Text.Encoding.UTF8.GetString(bytes.ToArray());
            return text.Length > 0 && text[0] == ' ' ? text.Substring(1) : text;
        }

        public int CountTokens(string text) => Encode(text).Count;

        /// <summary>tokenizer_sp.json 的结构，由 export_tokenizer_assets.py 生成。</summary>
        [Serializable]
        private class SpModelData
        {
            public string[] pieces = null;
            public float[] scores = null;
            public int[] types = null;
            public float min_score = 0f;
            public int unk_id = 0;
            public int bos_id = 1;
            public int eos_id = 2;
            public int pad_id = -1;
            public bool byte_fallback = true;
            public string normalizer_name = null;
            public bool add_dummy_prefix = true;
            public bool remove_extra_whitespaces = true;
            public bool escape_whitespaces = true;
            public bool treat_whitespace_as_suffix = false;
            public string charsmap_file = null;
        }
    }
}
