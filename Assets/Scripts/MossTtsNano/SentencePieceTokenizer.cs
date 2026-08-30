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

            Debug.Log($"[SentencePiece] Loaded {size} tokens, max piece length: {_maxPieceLength}");
        }

        /// <summary>
        /// 编码文本为 token IDs - 贪心最长匹配
        /// SentencePiece 在编码时在文本前自动添加 ▁ 前缀
        /// </summary>
        public List<int> Encode(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<int>();

            // SentencePiece 在开头自动添加空格（表示为 ▁）
            string prefixedText = "▁" + text;

            var tokenIds = new List<int>();
            int pos = 1; // 从 1 开始，跳过虚拟的 ▁ 前缀
            int n = prefixedText.Length;

            // 处理第一个片段（包含开头的 ▁）
            int bestLen = 0;
            int bestId = -1;
            float bestScore = float.MinValue;

            int maxLen = Math.Min(n, _maxPieceLength);
            for (int len = 1; len <= maxLen; len++)
            {
                string candidate = prefixedText.Substring(0, len);
                if (_pieceToId.TryGetValue(candidate, out int pieceId))
                {
                    float score = _idToScore[pieceId];
                    if (len > bestLen || (len == bestLen && score > bestScore))
                    {
                        bestLen = len;
                        bestId = pieceId;
                        bestScore = score;
                    }
                }
            }

            if (bestId >= 0)
            {
                tokenIds.Add(bestId);
                pos = bestLen;
            }
            else
            {
                tokenIds.Add(_unkId);
                pos = 1;
            }

            while (pos < n)
            {
                // 尝试匹配最长的片段
                bestLen = 0;
                bestId = -1;
                bestScore = float.MinValue;

                maxLen = Math.Min(n - pos, _maxPieceLength);
                for (int len = 1; len <= maxLen; len++)
                {
                    string candidate = prefixedText.Substring(pos, len);
                    if (_pieceToId.TryGetValue(candidate, out int pieceId))
                    {
                        float score = _idToScore[pieceId];
                        if (len > bestLen || (len == bestLen && score > bestScore))
                        {
                            bestLen = len;
                            bestId = pieceId;
                            bestScore = score;
                        }
                    }
                }

                if (bestId >= 0)
                {
                    tokenIds.Add(bestId);
                    pos += bestLen;
                }
                else
                {
                    // 无法匹配，使用 UNK
                    tokenIds.Add(_unkId);
                    pos += 1;
                }
            }

            return tokenIds;
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
