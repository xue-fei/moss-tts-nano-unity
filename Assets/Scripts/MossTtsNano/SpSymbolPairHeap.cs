using System;
using System.Collections.Generic;

namespace MossTtsNano
{
    /// <summary>
    /// BPE 合并用的最大堆，对应 bpe_model.cc 里
    /// std::priority_queue&lt;SymbolPair, ..., SymbolPairComparator&gt;。
    ///
    /// 排序语义必须与 SymbolPairComparator 完全一致：
    ///   score 大者优先；score 相同时 left 小者优先
    /// （原实现 `return score_less || (i1 == i2 &amp;&amp; h1.left > h2.left);`
    ///  表示 h1 更"小"、更晚出队，即 left 越小越先出队）。
    /// </summary>
    internal sealed class SpSymbolPairHeap
    {
        internal struct Pair
        {
            public float Score;
            public int Left;
            public int Right;
            public int Size;
        }

        private Pair[] _items;
        private int _count;

        public SpSymbolPairHeap(int capacity)
        {
            _items = new Pair[Math.Max(16, capacity)];
            _count = 0;
        }

        public int Count => _count;

        /// <summary>a 是否应排在 b 之前（即优先出队）。</summary>
        private static bool HasHigherPriority(in Pair a, in Pair b)
        {
            if (a.Score != b.Score) return a.Score > b.Score;
            return a.Left < b.Left;
        }

        public void Push(float score, int left, int right, int size)
        {
            if (_count == _items.Length)
                Array.Resize(ref _items, _items.Length << 1);

            _items[_count] = new Pair { Score = score, Left = left, Right = right, Size = size };
            int i = _count++;
            while (i > 0)
            {
                int parent = (i - 1) >> 1;
                if (!HasHigherPriority(_items[i], _items[parent])) break;
                (_items[i], _items[parent]) = (_items[parent], _items[i]);
                i = parent;
            }
        }

        public Pair Pop()
        {
            Pair top = _items[0];
            _count--;
            if (_count > 0)
            {
                _items[0] = _items[_count];
                int i = 0;
                while (true)
                {
                    int left = i * 2 + 1;
                    if (left >= _count) break;
                    int best = left;
                    int right = left + 1;
                    if (right < _count && HasHigherPriority(_items[right], _items[left]))
                        best = right;
                    if (!HasHigherPriority(_items[best], _items[i])) break;
                    (_items[i], _items[best]) = (_items[best], _items[i]);
                    i = best;
                }
            }
            return top;
        }

        public void Clear() => _count = 0;
    }
}
