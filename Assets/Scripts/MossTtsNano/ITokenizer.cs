using System.Collections.Generic;

namespace MossTtsNano
{
    /// <summary>
    /// 分词器接口 - 封装 SentencePiece 或类似分词逻辑
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>
        /// 编码文本为 token IDs
        /// </summary>
        List<int> Encode(string text);

        /// <summary>
        /// 解码 token IDs 为文本
        /// </summary>
        string Decode(List<int> tokenIds);

        /// <summary>
        /// 计算文本 token 数量
        /// </summary>
        int CountTokens(string text);
    }

    /// <summary>
    /// 简单 Unicode 分词器（占位实现）
    /// 实际使用时应替换为 SentencePiece 或其他分词器
    /// </summary>
    public class SimpleUnicodeTokenizer : ITokenizer
    {
        public List<int> Encode(string text)
        {
            var tokens = new List<int>();
            if (string.IsNullOrEmpty(text)) return tokens;

            foreach (char c in text)
                tokens.Add(c);

            return tokens;
        }

        public string Decode(List<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds)
                sb.Append((char)id);
            return sb.ToString();
        }

        public int CountTokens(string text)
        {
            return string.IsNullOrEmpty(text) ? 0 : text.Length;
        }
    }
}
