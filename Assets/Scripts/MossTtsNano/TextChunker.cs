using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace MossTtsNano
{
    /// <summary>
    /// 文本分块器 - 对应 Python 的 onnx_tts_runtime.py 中的 split_voice_clone_text 相关逻辑
    /// 用于将长文本按 token 预算分割成适合语音合成的片段
    /// </summary>
    public static class TextChunker
    {
        private static readonly char[] SentenceEndPunctuation = { '.', '!', '?', '。', '！', '？', '；', ';' };
        private static readonly char[] ClauseSplitPunctuation = { ',', '，', '、', '；', ';', '：', ':' };
        private static readonly char[] ClosingPunctuation = { '"', '\'', '”', '’', ')', ']', '）', '】', '》', '」', '』' };

        private static readonly HashSet<char> SentenceEndSet = new HashSet<char>(SentenceEndPunctuation);
        private static readonly HashSet<char> ClauseSplitSet = new HashSet<char>(ClauseSplitPunctuation);
        private static readonly HashSet<char> ClosingSet = new HashSet<char>(ClosingPunctuation);
        private static readonly HashSet<char> PreferredBoundarySet = new HashSet<char>();

        static TextChunker()
        {
            PreferredBoundarySet.UnionWith(ClauseSplitPunctuation);
            PreferredBoundarySet.UnionWith(SentenceEndPunctuation);
            PreferredBoundarySet.Add(' ');
        }

        private const float DefaultInterChunkPauseShortSec = 0.40f;
        private const float DefaultInterChunkPauseLongSec = 0.24f;

        /// <summary>
        /// 检测是否包含 CJK 字符
        /// </summary>
        public static bool ContainsCjk(string text)
        {
            if (string.IsNullOrEmpty(text)) return false;
            foreach (char c in text)
            {
                if ((c >= '\u4e00' && c <= '\u9fff') ||
                    (c >= '\u3400' && c <= '\u4dbf') ||
                    (c >= '\u3040' && c <= '\u30ff') ||
                    (c >= '\uac00' && c <= '\ud7af'))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// 预处理文本用于分句
        /// </summary>
        public static string PrepareForSentenceChunking(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text prompt cannot be empty.");

            string normalized = text.Trim().Replace("\r", " ").Replace("\n", " ");
            while (normalized.Contains("  "))
                normalized = normalized.Replace("  ", " ");

            if (ContainsCjk(normalized))
            {
                char last = normalized[normalized.Length - 1];
                if (!SentenceEndSet.Contains(last))
                    normalized += "。";
                return normalized;
            }

            if (char.IsLower(normalized[0]))
                normalized = char.ToUpper(normalized[0]) + normalized.Substring(1);

            if (char.IsLetterOrDigit(normalized[normalized.Length - 1]))
                normalized += ".";

            if (normalized.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Length < 5)
                normalized = "        " + normalized;

            return normalized;
        }

        /// <summary>
        /// 按标点符号切分文本
        /// </summary>
        public static List<string> SplitByPunctuation(string text, HashSet<char> punctuation)
        {
            var sentences = new List<string>();
            var current = new StringBuilder();
            int index = 0;
            string normalized = text ?? "";

            while (index < normalized.Length)
            {
                char c = normalized[index];
                current.Append(c);

                if (punctuation.Contains(c))
                {
                    // 收集尾部闭合标点
                    int lookahead = index + 1;
                    while (lookahead < normalized.Length && ClosingSet.Contains(normalized[lookahead]))
                    {
                        current.Append(normalized[lookahead]);
                        lookahead++;
                    }

                    string sentence = current.ToString().Trim();
                    if (!string.IsNullOrEmpty(sentence))
                        sentences.Add(sentence);

                    current.Clear();

                    while (lookahead < normalized.Length && char.IsWhiteSpace(normalized[lookahead]))
                        lookahead++;

                    index = lookahead;
                    continue;
                }
                index++;
            }

            string tail = current.ToString().Trim();
            if (!string.IsNullOrEmpty(tail))
                sentences.Add(tail);

            return sentences;
        }

        /// <summary>
        /// 合并两个句子片段
        /// </summary>
        public static string JoinSentenceParts(string left, string right)
        {
            if (string.IsNullOrEmpty(left)) return right;
            if (string.IsNullOrEmpty(right)) return left;
            if (ContainsCjk(left) || ContainsCjk(right))
                return left + right;
            return left + " " + right;
        }

        /// <summary>
        /// 按 token 预算切分文本（二分查找最优切分点）
        /// </summary>
        public static List<string> SplitByTokenBudget(string text, int maxTokens, Func<string, int> countTokens)
        {
            var pieces = new List<string>();
            string remaining = text?.Trim();
            if (string.IsNullOrEmpty(remaining)) return pieces;

            while (!string.IsNullOrEmpty(remaining))
            {
                if (countTokens(remaining) <= maxTokens)
                {
                    pieces.Add(remaining);
                    break;
                }

                int low = 1, high = remaining.Length;
                int bestPrefixLength = 1;

                while (low <= high)
                {
                    int mid = (low + high) / 2;
                    string candidate = remaining.Substring(0, mid).Trim();
                    if (string.IsNullOrEmpty(candidate))
                    {
                        low = mid + 1;
                        continue;
                    }

                    if (countTokens(candidate) <= maxTokens)
                    {
                        bestPrefixLength = mid;
                        low = mid + 1;
                    }
                    else
                    {
                        high = mid - 1;
                    }
                }

                int cutIndex = bestPrefixLength;
                string prefix = remaining.Substring(0, bestPrefixLength);
                int preferredIndex = -1;
                int scanMin = Math.Max(-1, prefix.Length - 25);

                for (int scanIdx = prefix.Length - 1; scanIdx > scanMin; scanIdx--)
                {
                    if (PreferredBoundarySet.Contains(prefix[scanIdx]))
                    {
                        preferredIndex = scanIdx + 1;
                        break;
                    }
                }

                if (preferredIndex > 0)
                    cutIndex = preferredIndex;

                string piece = remaining.Substring(0, cutIndex).Trim();
                if (string.IsNullOrEmpty(piece))
                {
                    piece = remaining.Substring(0, bestPrefixLength).Trim();
                    cutIndex = bestPrefixLength;
                }

                pieces.Add(piece);
                remaining = remaining.Substring(cutIndex).Trim();
            }

            return pieces;
        }

        /// <summary>
        /// 语音克隆文本分块 - 主入口
        /// </summary>
        public static List<string> SplitVoiceCloneText(string text, int maxTokens, Func<string, int> countTokens)
        {
            string normalizedText = text?.Trim();
            if (string.IsNullOrEmpty(normalizedText)) return new List<string>();

            int safeMaxTokens = Math.Max(1, maxTokens);
            string preparedText = PrepareForSentenceChunking(normalizedText);
            List<string> sentenceCandidates = SplitByPunctuation(preparedText, SentenceEndSet);
            if (sentenceCandidates == null || sentenceCandidates.Count == 0)
                sentenceCandidates = new List<string> { preparedText.Trim() };

            var sentenceSlices = new List<(int tokenCount, string text)>();

            foreach (string sentenceText in sentenceCandidates)
            {
                string normalizedSentence = sentenceText.Trim();
                if (string.IsNullOrEmpty(normalizedSentence)) continue;

                int sentenceTokenCount = countTokens(normalizedSentence);
                if (sentenceTokenCount <= safeMaxTokens)
                {
                    sentenceSlices.Add((sentenceTokenCount, normalizedSentence));
                    continue;
                }

                // 长句按子句切分
                List<string> clauseCandidates = SplitByPunctuation(normalizedSentence, ClauseSplitSet);
                if (clauseCandidates == null || clauseCandidates.Count <= 1)
                    clauseCandidates = new List<string> { normalizedSentence };

                foreach (string clauseText in clauseCandidates)
                {
                    string normalizedClause = clauseText.Trim();
                    if (string.IsNullOrEmpty(normalizedClause)) continue;

                    int clauseTokenCount = countTokens(normalizedClause);
                    if (clauseTokenCount <= safeMaxTokens)
                    {
                        sentenceSlices.Add((clauseTokenCount, normalizedClause));
                        continue;
                    }

                foreach (string piece in SplitByTokenBudget(normalizedClause, safeMaxTokens, countTokens))
                    {
                        string normalizedPiece = piece.Trim();
                        if (!string.IsNullOrEmpty(normalizedPiece))
                            sentenceSlices.Add((countTokens(normalizedPiece), normalizedPiece));
                    }
                }
            }

            // 贪心合并到 chunks
            var chunks = new List<string>();
            string currentChunk = "";
            int currentChunkTokenCount = 0;

            foreach (var (sentenceTokenCount, sentenceText) in sentenceSlices)
            {
                if (string.IsNullOrEmpty(currentChunk))
                {
                    currentChunk = sentenceText;
                    currentChunkTokenCount = sentenceTokenCount;
                    continue;
                }

                if (currentChunkTokenCount + sentenceTokenCount > safeMaxTokens)
                {
                    chunks.Add(currentChunk.Trim());
                    currentChunk = sentenceText;
                    currentChunkTokenCount = sentenceTokenCount;
                }
                else
                {
                    currentChunk = JoinSentenceParts(currentChunk, sentenceText);
                    currentChunkTokenCount = countTokens(currentChunk);
                }
            }

            if (!string.IsNullOrEmpty(currentChunk))
                chunks.Add(currentChunk.Trim());

            return chunks.Count > 1 ? chunks : new List<string> { normalizedText };
        }

        /// <summary>
        /// 估算块间停顿秒数
        /// </summary>
        public static float EstimateInterChunkPauseSeconds(string textChunk)
        {
            int wordCount = textChunk?.Trim().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Length ?? 0;
            return wordCount <= 4 ? DefaultInterChunkPauseShortSec : DefaultInterChunkPauseLongSec;
        }
    }
}
