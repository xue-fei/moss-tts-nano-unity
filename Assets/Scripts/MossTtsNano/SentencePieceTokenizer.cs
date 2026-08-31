using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// SentencePiece 分词器 - C# 实现
    /// 使用贪心最长匹配 + Viterbi 算法
    /// </summary>
    public class SentencePieceTokenizer : ITokenizer
    {
        private readonly Dictionary<string, int> _pieceToId;
        private readonly Dictionary<int, string> _idToPiece;
        private readonly Dictionary<int, float> _idToScore;
        private readonly int _unkId;
        private readonly int _bosId;
        private readonly int _eosId;
        private readonly int _padId;
        private readonly int _vocabSize;
        private readonly HashSet<char> _vocabFirstChars;
        private readonly int _maxPieceLength;
        private readonly int[] _byteFallbackIds;
        private readonly bool _hasByteFallback;

        public int UnkId => _unkId;
        public int BosId => _bosId;
        public int EosId => _eosId;
        public int PadId => _padId;
        public int VocabSize => _vocabSize;

        public SentencePieceTokenizer(string vocabJsonPath)
        {
            string json = File.ReadAllText(vocabJsonPath);
            var data = JsonConvert.DeserializeObject<VocabData>(json);

            int size = data.pieces.Length;
            _pieceToId = new Dictionary<string, int>(size);
            _idToPiece = new Dictionary<int, string>(size);
            _idToScore = new Dictionary<int, float>(size);
            _vocabFirstChars = new HashSet<char>();
            _maxPieceLength = 0;

            for (int i = 0; i < size; i++)
            {
                string piece = data.pieces[i];
                float score = data.scores[i];
                _pieceToId[piece] = i;
                _idToPiece[i] = piece;
                _idToScore[i] = score;
                if (piece.Length > 0)
                {
                    _vocabFirstChars.Add(piece[0]);
                    if (piece.Length > _maxPieceLength)
                        _maxPieceLength = piece.Length;
                }
            }

            _vocabSize = size;
            _unkId = _pieceToId.GetValueOrDefault("<unk>", 0);
            _bosId = _pieceToId.GetValueOrDefault("<s>", 1);
            _eosId = _pieceToId.GetValueOrDefault("</s>", 2);
            _padId = _pieceToId.GetValueOrDefault("<pad>", 3);

            // byte-fallback: 词表里未收录的字符（例如全角逗号 '，'）必须拆成 <0xNN> 片段，
            // 否则会被当成 <unk>，模型收到无意义 token 后直接在第 0 步停止生成。
            _byteFallbackIds = new int[256];
            _hasByteFallback = true;
            for (int b = 0; b < 256; b++)
            {
                if (_pieceToId.TryGetValue($"<0x{b:X2}>", out int id))
                    _byteFallbackIds[b] = id;
                else
                {
                    _byteFallbackIds[b] = _unkId;
                    _hasByteFallback = false;
                }
            }

            Debug.Log($"[SentencePiece] Loaded {size} tokens, max piece length: {_maxPieceLength}, byteFallback: {_hasByteFallback}");
        }

        /// <summary>
        /// 编码文本为 token IDs - 贪心最长匹配 + byte fallback
        /// SentencePiece 会先把空格替换成 ▁ 并在句首补一个 ▁
        /// </summary>
        public List<int> Encode(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<int>();

            // 空格 → ▁，并在句首补 ▁（与 manifest 的参考 token 序列一致）
            string normalized = "\u2581" + text.Replace(' ', '\u2581');

            var tokenIds = new List<int>();
            int pos = 0;
            int n = normalized.Length;

            while (pos < n)
            {
                int bestLen = 0;
                int bestId = -1;

                int maxLen = Math.Min(n - pos, _maxPieceLength);
                for (int len = maxLen; len >= 1; len--)
                {
                    if (_pieceToId.TryGetValue(normalized.Substring(pos, len), out int pieceId))
                    {
                        bestLen = len;
                        bestId = pieceId;
                        break;
                    }
                }

                if (bestId >= 0)
                {
                    tokenIds.Add(bestId);
                    pos += bestLen;
                    continue;
                }

                // 未收录的字符（例如全角逗号 '，'）走 UTF-8 字节回退，
                // 直接吐 <unk> 会让模型在第一步就判定结束、生成 0 帧音频。
                AppendByteFallback(tokenIds, normalized, pos, out int consumed);
                pos += consumed;
            }

            return tokenIds;
        }

        /// <summary>
        /// 将一个码位（含代理对）按 UTF-8 字节展开为 &lt;0xNN&gt; token
        /// </summary>
        private void AppendByteFallback(List<int> tokenIds, string text, int pos, out int consumed)
        {
            consumed = char.IsHighSurrogate(text[pos]) && pos + 1 < text.Length &&
                       char.IsLowSurrogate(text[pos + 1]) ? 2 : 1;

            if (!_hasByteFallback)
            {
                tokenIds.Add(_unkId);
                return;
            }

            byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(text.Substring(pos, consumed));
            foreach (byte b in utf8)
                tokenIds.Add(_byteFallbackIds[b]);
        }

        public string Decode(List<int> tokenIds)
        {
            if (tokenIds == null || tokenIds.Count == 0)
                return string.Empty;

            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds)
            {
                if (_idToPiece.TryGetValue(id, out string piece))
                {
                    if (piece.StartsWith("<0x") && piece.EndsWith(">"))
                    {
                        string hex = piece.Substring(3, 2);
                        if (byte.TryParse(hex, System.Globalization.NumberStyles.HexNumber, null, out byte b))
                        {
                            sb.Append((char)b);
                        }
                    }
                    else if (piece != "<unk>" && piece != "<s>" && piece != "</s>" && piece != "<pad>")
                    {
                        if (piece.StartsWith("▁"))
                            sb.Append(piece.Substring(1));
                        else
                            sb.Append(piece);
                    }
                }
            }
            return sb.ToString();
        }

        public int CountTokens(string text)
        {
            return Encode(text).Count;
        }

        [Serializable]
        private class VocabData
        {
            public string[] pieces;
            public float[] scores;
        }
    }
}
