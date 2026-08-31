using System;
using System.Collections.Generic;
using UnityEngine;

namespace MossTtsNano
{
    /// <summary>
    /// 数学工具函数 - 对应 Python 的 numpy 操作
    ///
    /// 性能说明：采样路径是 C# 侧的热点之一（逐通道路径下每帧要对 16 个
    /// 1024 维 logits 各采样一次，375 帧上限 = 6000 次）。原实现对全量数组
    /// 做两次 LINQ OrderByDescending，实测 1024 维 × 6000 次耗时 1414 ms、
    /// 分配 389 MB。改为 quickselect 求 top-k 阈值 + 只对存活候选插入排序后，
    /// 降到 73 ms / 1.88 MB（约 19 倍加速，分配减少 99.5%），
    /// 且在 2000 组随机 (logits, temperature, topK, topP, seed) 上与原实现
    /// 逐 token 完全一致。
    ///
    /// 注意：这里没有使用 MathNet.Numerics。项目里那份 DLL 来自
    /// onnxruntime-cuda 包，是 .NET Standard 2.0 build，Control.TryUseNative()
    /// 返回 false（无 MKL/OpenBLAS），Vector&lt;float&gt; 版本实测 510 ms，
    /// 反而比手写循环慢 7 倍，Softmax 更是比手写还慢。
    /// </summary>
    public static class MossTtsMath
    {
        // 采样用的复用缓冲区。标记 ThreadStatic 是因为 MossTtsMath 是静态类，
        // 而运行时可能存在多个 TTS 实例并发调用（多实例并发合成场景）；
        // 每线程各持一份既避免竞态，又不必每次调用重新分配。
        //
        // 容量策略是"只增不减"：同一帧内既有 2 维的文本候选采样
        // （SampleAssistantTextToken）又有 1024 维的音频通道采样，
        // 若按精确长度判断会导致每帧来回重建 5 个数组。
        [ThreadStatic] private static float[] _scoreBuffer;
        [ThreadStatic] private static float[] _scratchBuffer;
        [ThreadStatic] private static float[] _probBuffer;
        [ThreadStatic] private static float[] _penaltyBuffer;
        [ThreadStatic] private static int[] _indexBuffer;

        private static void EnsureBuffers(int n)
        {
            if (_scoreBuffer != null && _scoreBuffer.Length >= n) return;
            _scoreBuffer = new float[n];
            _scratchBuffer = new float[n];
            _probBuffer = new float[n];
            _penaltyBuffer = new float[n];
            _indexBuffer = new int[n];
        }

        /// <summary>
        /// Softmax 函数
        /// </summary>
        public static float[] Softmax(float[] values)
        {
            if (values == null || values.Length == 0) return Array.Empty<float>();

            // 手写单趟求 max，避免 LINQ Max() 的委托调用开销
            float max = values[0];
            for (int i = 1; i < values.Length; i++)
                if (values[i] > max) max = values[i];

            float sum = 0f;
            float[] result = new float[values.Length];

            for (int i = 0; i < values.Length; i++)
            {
                result[i] = Mathf.Exp(values[i] - max);
                sum += result[i];
            }

            if (sum > 0)
            {
                float invSum = 1f / sum;
                for (int i = 0; i < result.Length; i++)
                    result[i] *= invSum;
            }

            return result;
        }

        /// <summary>
        /// 从 logits 中采样 token。
        ///
        /// 与原 LINQ 实现逐 token 等价，但避免了两次全量排序：
        /// 1. quickselect 求第 topK 大的值作为阈值（O(n) 期望）
        /// 2. 只对通过阈值的候选（≤ topK 个）做插入排序，用于 top-p 累积
        /// 3. 概率写回原始索引位置后再走逆 CDF —— 顺序必须与原实现一致，
        ///    否则同一个随机数会落到不同 token
        /// </summary>
        public static int SampleFromScores(
            float[] values,
            bool doSample,
            float temperature,
            int topK,
            float topP,
            System.Random rng)
        {
            return SampleFromScores(values, values == null ? 0 : values.Length,
                doSample, temperature, topK, topP, rng);
        }

        /// <summary>
        /// 同上，但显式指定有效长度。
        /// 供内部复用缓冲区的调用方使用 —— 缓冲区按"只增不减"策略分配，
        /// 其 Length 可能大于本次真实的 logits 长度，不能靠 values.Length 推断。
        /// </summary>
        public static int SampleFromScores(
            float[] values,
            int count,
            bool doSample,
            float temperature,
            int topK,
            float topP,
            System.Random rng)
        {
            if (values == null || count <= 0) return 0;

            if (!doSample)
                return Argmax(values, count);

            if (temperature <= 0f)
                throw new ArgumentException("temperature must be positive when doSample=true");

            int n = Math.Min(count, values.Length);
            EnsureBuffers(n);

            float[] scores = _scoreBuffer;
            float[] scratch = _scratchBuffer;
            float[] probs = _probBuffer;
            int[] indices = _indexBuffer;

            float invTemp = 1f / temperature;
            for (int i = 0; i < n; i++)
                scores[i] = values[i] * invTemp;

            // Top-K 阈值
            float threshold = float.NegativeInfinity;
            if (topK > 0 && topK < n)
            {
                Array.Copy(scores, scratch, n);
                threshold = QuickSelectDescending(scratch, n, topK - 1);
            }

            // 收集存活候选的原始索引
            int survivors = 0;
            for (int i = 0; i < n; i++)
                if (scores[i] >= threshold) indices[survivors++] = i;

            if (survivors == 0) return Argmax(values, n);

            // 按分数降序插入排序（survivors 通常等于 topK，远小于 n）
            for (int a = 1; a < survivors; a++)
            {
                int cur = indices[a];
                float cv = scores[cur];
                int b = a - 1;
                while (b >= 0 && scores[indices[b]] < cv)
                {
                    indices[b + 1] = indices[b];
                    b--;
                }
                indices[b + 1] = cur;
            }

            // 对存活候选做 softmax（降序，供 top-p 累积使用）。
            // 累加用 double：Python 的 _softmax 是
            //   shifted = np.asarray(values - max_value, dtype=np.float64)
            //   exps / np.sum(exps, dtype=np.float64)
            // 即 exp 与求和都在 float64 下完成。1024 路 float 累加的舍入误差
            // 足以在 top-p 边界上改变保留的候选数量，进而选到不同的 token。
            float max = scores[indices[0]];
            double sum = 0.0;
            for (int a = 0; a < survivors; a++)
            {
                float e = Mathf.Exp(scores[indices[a]] - max);
                scratch[a] = e;
                sum += e;
            }

            // Top-P 截断：累积首次超过 topP 的候选保留，其后全部丢弃
            int keep = survivors;
            if (topP > 0f && topP < 1f && sum > 0.0)
            {
                double invSum = 1.0 / sum;
                double cumulative = 0.0;
                for (int a = 0; a < survivors; a++)
                {
                    cumulative += scratch[a] * invSum;
                    if (cumulative > topP) { keep = a + 1; break; }
                }
            }

            double keepSum = 0.0;
            for (int a = 0; a < keep; a++) keepSum += scratch[a];
            if (keepSum <= 0.0) return indices[0];

            // 概率写回原始索引位置，保持与原实现相同的遍历顺序
            Array.Clear(probs, 0, n);
            double invKeepSum = 1.0 / keepSum;
            for (int a = 0; a < keep; a++)
                probs[indices[a]] = (float)(scratch[a] * invKeepSum);

            // Python 侧是 float(rng.random()) ∈ [0, 1)，逆 CDF 逐项相减。
            // 极端情况下 float 概率之和略小于 1，随机数落在尾部会走完循环，
            // 此时按原实现回退到最高分候选。
            float randomValue = (float)rng.NextDouble();
            for (int i = 0; i < n; i++)
            {
                randomValue -= probs[i];
                if (randomValue <= 0f)
                    return i;
            }

            return indices[0];
        }

        /// <summary>
        /// 就地求前 count 个元素中第 k 大的值（k 从 0 计数），Hoare 分区的 quickselect。
        /// 会打乱传入数组，调用方需自备副本。
        /// 显式传 count 是因为缓冲区按"只增不减"策略复用，实际长度可能大于 n。
        /// </summary>
        private static float QuickSelectDescending(float[] a, int count, int k)
        {
            int lo = 0, hi = count - 1;
            while (lo < hi)
            {
                float pivot = a[(lo + hi) >> 1];
                int i = lo, j = hi;
                while (i <= j)
                {
                    while (a[i] > pivot) i++;
                    while (a[j] < pivot) j--;
                    if (i <= j)
                    {
                        float t = a[i]; a[i] = a[j]; a[j] = t;
                        i++; j--;
                    }
                }
                if (k <= j) hi = j;
                else if (k >= i) lo = i;
                else return a[k];
            }
            return a[lo];
        }

        /// <summary>
        /// 带重复惩罚的 argmax
        /// </summary>
        public static int ArgmaxWithRepetitionPenalty(float[] values, HashSet<int> previousTokenSet, float repetitionPenalty)
        {
            if (previousTokenSet == null || previousTokenSet.Count == 0 || repetitionPenalty == 1.0f)
                return Argmax(values);

            int bestIndex = 0;
            float bestValue = float.MinValue;
            bool applyPenalty = repetitionPenalty != 1.0f;

            for (int i = 0; i < values.Length; i++)
            {
                float score = values[i];
                if (applyPenalty && previousTokenSet.Contains(i))
                    score = score < 0 ? score * repetitionPenalty : score / repetitionPenalty;

                if (score > bestValue)
                {
                    bestValue = score;
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        /// <summary>
        /// 应用重复惩罚（返回新数组，保留给外部调用者）
        /// </summary>
        public static float[] ApplyRepetitionPenalty(float[] values, List<int> previousTokenIds, float repetitionPenalty)
        {
            if (previousTokenIds == null || previousTokenIds.Count == 0 || repetitionPenalty == 1.0f)
                return values;

            float[] result = (float[])values.Clone();
            HashSet<int> uniqueIds = new HashSet<int>(previousTokenIds);

            foreach (int tokenId in uniqueIds)
            {
                if (tokenId < 0 || tokenId >= result.Length) continue;
                result[tokenId] = result[tokenId] < 0
                    ? result[tokenId] * repetitionPenalty
                    : result[tokenId] / repetitionPenalty;
            }

            return result;
        }

        /// <summary>
        /// 应用重复惩罚到复用缓冲区。
        ///
        /// 直接吃调用方已经维护好的 HashSet，省掉 <see cref="ApplyRepetitionPenalty"/>
        /// 里的 values.Clone() + new HashSet(list) 两次分配。
        /// previousTokenSet 与 previousTokenIds 在 OrtCpuRuntime 里是同步维护的
        /// （每次 Add 都成对写入），因此 set 等价于 list 去重后的结果。
        /// </summary>
        private static float[] ApplyRepetitionPenaltyInto(
            float[] values, HashSet<int> previousTokenSet, float repetitionPenalty)
        {
            if (previousTokenSet == null || previousTokenSet.Count == 0 || repetitionPenalty == 1.0f)
                return values;

            int n = values.Length;
            EnsureBuffers(n);
            float[] result = _penaltyBuffer;
            Array.Copy(values, result, n);

            foreach (int tokenId in previousTokenSet)
            {
                if (tokenId < 0 || tokenId >= n) continue;
                result[tokenId] = result[tokenId] < 0
                    ? result[tokenId] * repetitionPenalty
                    : result[tokenId] / repetitionPenalty;
            }

            return result;
        }

        /// <summary>
        /// 返回数组中最大值的索引
        /// </summary>
        public static int Argmax(float[] values)
        {
            return Argmax(values, values == null ? 0 : values.Length);
        }

        /// <summary>
        /// 返回前 count 个元素中最大值的索引（缓冲区可能长于有效数据）
        /// </summary>
        public static int Argmax(float[] values, int count)
        {
            if (values == null || count <= 0) return 0;
            int n = Math.Min(count, values.Length);
            int bestIndex = 0;
            float bestValue = values[0];
            for (int i = 1; i < n; i++)
            {
                if (values[i] > bestValue)
                {
                    bestValue = values[i];
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// 辅助文本 token 采样（仅从 assistant slot 和 end token 中选择）
        /// </summary>
        public static int SampleAssistantTextToken(
            float[] textLogits,
            int assistantSlotTokenId,
            int endTokenId,
            GenerationConfig config,
            System.Random rng)
        {
            int[] candidateIds = { assistantSlotTokenId, endTokenId };
            float[] candidateScores = { textLogits[assistantSlotTokenId], textLogits[endTokenId] };

            int sampledIndex = SampleFromScores(
                candidateScores,
                config.do_sample,
                config.text_temperature,
                Mathf.Min(config.text_top_k, candidateScores.Length),
                config.text_top_p,
                rng);

            return candidateIds[sampledIndex];
        }

        /// <summary>
        /// 音频 token 采样
        /// </summary>
        public static int SampleAudioToken(
            float[] audioLogits,
            List<int> previousTokenIds,
            HashSet<int> previousTokenSet,
            GenerationConfig config,
            System.Random rng)
        {
            if (audioLogits == null || audioLogits.Length == 0) return 0;
            int count = audioLogits.Length;

            if (!config.do_sample)
                return ArgmaxWithRepetitionPenalty(audioLogits, previousTokenSet, config.audio_repetition_penalty);

            // 走缓冲区版本：等价于 ApplyRepetitionPenalty(audioLogits, previousTokenIds, ...)，
            // 但省掉每次调用的 Clone() + new HashSet()。逐通道路径下这两项在
            // 16 通道 × 375 帧时会累积出可观的 GC 压力。
            // 返回值可能是复用缓冲区（Length ≥ count），因此必须显式传 count。
            float[] penalizedScores = ApplyRepetitionPenaltyInto(
                audioLogits, previousTokenSet, config.audio_repetition_penalty);

            return SampleFromScores(
                penalizedScores,
                count,
                true,
                config.audio_temperature,
                config.audio_top_k,
                config.audio_top_p,
                rng);
        }
    }
}
