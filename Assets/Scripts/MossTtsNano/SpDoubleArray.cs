using System;

namespace MossTtsNano
{
    /// <summary>
    /// darts_clone DoubleArray 的只读实现，对应
    /// sentencepiece/third_party/darts_clone/darts.h 中的 DoubleArrayImpl。
    ///
    /// SentencePiece 的 nmt_nfkc 归一化规则表（precompiled_charsmap）就是一个
    /// 序列化好的 DoubleArray trie，这里必须按位复刻它的取值方式，否则
    /// 全角标点不会被折叠成半角，模型会收到 byte-fallback 的无语义 token。
    ///
    /// 每个 unit 是一个 little-endian uint32，位域含义：
    ///   bit 8       has_leaf —— 是否直接派生出叶子节点
    ///   bit 0..7    label    —— 转移标签（叶子节点的 MSB 为 1，故 label 无效）
    ///   bit 9       offset 的编码模式（决定左移 0 位还是 8 位）
    ///   bit 0..30   value    —— 仅叶子节点有效
    /// </summary>
    internal sealed class SpDoubleArray
    {
        private readonly uint[] _units;

        public SpDoubleArray(byte[] blob, int offset, int length)
        {
            int count = length / 4;
            _units = new uint[count];
            Buffer.BlockCopy(blob, offset, _units, 0, count * 4);
        }

        private static bool HasLeaf(uint unit) => ((unit >> 8) & 1) == 1;

        private static uint Value(uint unit) => unit & ((1u << 31) - 1);

        private static uint Label(uint unit) => unit & ((1u << 31) | 0xFF);

        private static uint Offset(uint unit) => (unit >> 10) << (int)((unit & (1u << 9)) >> 6);

        /// <summary>
        /// commonPrefixSearch 的「只取最长命中」特化版本。
        ///
        /// Normalizer::NormalizePrefix 遍历所有命中只为挑出最长的那条规则
        /// （`longest_length == 0 || length > longest_length`），
        /// 所以这里边遍历边取最长，避免分配结果数组。
        /// </summary>
        /// <returns>命中时返回 true，并输出规则的 value 与匹配字节长度。</returns>
        public bool LongestPrefix(byte[] key, int start, int end, out uint value, out int length)
        {
            value = 0;
            length = 0;

            uint nodePos = 0;
            uint unit = _units[nodePos];
            nodePos ^= Offset(unit);

            for (int i = start; i < end; i++)
            {
                nodePos ^= key[i];
                unit = _units[nodePos];
                if (Label(unit) != key[i])
                    break;

                nodePos ^= Offset(unit);
                if (HasLeaf(unit))
                {
                    // 后命中的一定更长，直接覆盖即可（等价于原实现的 > 比较）。
                    value = Value(_units[nodePos]);
                    length = i - start + 1;
                }
            }

            return length > 0;
        }
    }
}
