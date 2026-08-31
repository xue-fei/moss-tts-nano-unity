using System;
using System.Collections.Generic;

namespace MossTtsNano
{
    /// <summary>
    /// 以「字节区间」为键的开放寻址哈希表。
    ///
    /// BPE 合并每一步都要用 left+right 拼出的字节串去查词表 id。
    /// 若用 Dictionary&lt;string, int&gt;，每次查询都要先把字节转成 string，
    /// 在一句话上会产生上千次临时分配。这里直接对
    /// (buffer, start, length) 求哈希并比较，查询过程零分配。
    /// </summary>
    internal sealed class SpByteMap
    {
        private byte[][] _keys;
        private int[] _values;
        private int _mask;
        private int _count;

        public SpByteMap(int capacityHint)
        {
            int capacity = 16;
            while (capacity < capacityHint * 2) capacity <<= 1;
            _keys = new byte[capacity][];
            _values = new int[capacity];
            _mask = capacity - 1;
        }

        private static int Hash(byte[] buf, int start, int length)
        {
            // FNV-1a 32 位
            uint hash = 2166136261u;
            int end = start + length;
            for (int i = start; i < end; i++)
            {
                hash ^= buf[i];
                hash *= 16777619u;
            }
            return (int)(hash & 0x7FFFFFFF);
        }

        private static bool KeyEquals(byte[] key, byte[] buf, int start, int length)
        {
            if (key.Length != length) return false;
            for (int i = 0; i < length; i++)
                if (key[i] != buf[start + i]) return false;
            return true;
        }

        public void Add(byte[] key, int value)
        {
            if (_count * 2 >= _keys.Length) Grow();

            int slot = Hash(key, 0, key.Length) & _mask;
            while (_keys[slot] != null)
            {
                if (KeyEquals(_keys[slot], key, 0, key.Length))
                {
                    _values[slot] = value;
                    return;
                }
                slot = (slot + 1) & _mask;
            }
            _keys[slot] = key;
            _values[slot] = value;
            _count++;
        }

        public bool TryGet(byte[] buf, int start, int length, out int value)
        {
            int slot = Hash(buf, start, length) & _mask;
            while (_keys[slot] != null)
            {
                if (KeyEquals(_keys[slot], buf, start, length))
                {
                    value = _values[slot];
                    return true;
                }
                slot = (slot + 1) & _mask;
            }
            value = -1;
            return false;
        }

        private void Grow()
        {
            byte[][] oldKeys = _keys;
            int[] oldValues = _values;

            int capacity = oldKeys.Length << 1;
            _keys = new byte[capacity][];
            _values = new int[capacity];
            _mask = capacity - 1;
            _count = 0;

            for (int i = 0; i < oldKeys.Length; i++)
                if (oldKeys[i] != null) Add(oldKeys[i], oldValues[i]);
        }
    }
}
